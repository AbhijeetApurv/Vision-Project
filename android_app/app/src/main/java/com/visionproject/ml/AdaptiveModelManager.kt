package com.visionproject.ml

import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.security.MessageDigest
import java.util.concurrent.locks.ReentrantReadWriteLock
import kotlin.concurrent.read
import kotlin.concurrent.write

/**
 * PLANE 2: 3-Tier model storage manager with SHA-256 verification and
 * thread-safe ReadWriteLock hot-swapping for zero-drop CameraX frame analysis.
 *
 * Storage Tiers:
 * - Tier 3 (Active OTA): filesDir/ota/active_model.tflite
 * - Tier 2 (Previous Verified Rollback): filesDir/ota/rollback_model.tflite
 * - Tier 1 (Factory Bundled Asset Fallback): assets/yolov8n_int8_factory.tflite
 */
class AdaptiveModelManager(private val context: Context) {

    data class Heuristics(
        val confidenceThreshold: Float = 0.45f,
        val iouThreshold: Float = 0.50f,
        val cooldownMs: Long = 3500L,
        val criticalAreaRatio: Float = 0.18f
    )

    private val rwLock = ReentrantReadWriteLock()
    private val otaDir = File(context.filesDir, "ota").apply { mkdirs() }
    private val activeModelFile = File(otaDir, "active_model.tflite")
    private val rollbackModelFile = File(otaDir, "rollback_model.tflite")
    private val activeManifestFile = File(otaDir, "manifest.json")

    @Volatile
    var currentHeuristics: Heuristics = Heuristics()
        private set

    private var activeModelBuffer: ByteBuffer? = null

    init {
        loadBestAvailableTier()
    }

    /**
     * Executes inference safely under the read lock so background OTA swaps
     * never corrupt or close an in-flight interpreter buffer.
     */
    fun <T> withModelBuffer(block: (ByteBuffer, Heuristics) -> T): T? {
        return rwLock.read {
            val buffer = activeModelBuffer ?: return@read null
            block(buffer.duplicate(), currentHeuristics)
        }
    }

    /**
     * Atomically verifies and promotes a newly downloaded OTA model to Tier 3,
     * rotating the previous Tier 3 model into Tier 2 (rollback).
     */
    fun applyOtaUpdate(stagedModelFile: File, expectedSha256: String, manifestJson: String): Boolean {
        val actualHash = computeSha256(stagedModelFile)
        if (!actualHash.equals(expectedSha256, ignoreCase = true)) {
            Log.e(TAG, "OTA SHA-256 mismatch! Expected=$expectedSha256, Actual=$actualHash")
            stagedModelFile.delete()
            return false
        }

        val parsedHeuristics = parseHeuristics(manifestJson)
        val newBuffer = mapFileToReadOnlyBuffer(stagedModelFile) ?: return false

        rwLock.write {
            if (activeModelFile.exists()) {
                activeModelFile.copyTo(rollbackModelFile, overwrite = true)
            }
            stagedModelFile.copyTo(activeModelFile, overwrite = true)
            stagedModelFile.delete()
            activeManifestFile.writeText(manifestJson)

            activeModelBuffer = newBuffer
            currentHeuristics = parsedHeuristics
            Log.i(TAG, "Successfully hot-swapped to Tier-3 OTA model.")
        }
        return true
    }

    private fun loadBestAvailableTier() {
        rwLock.write {
            // Tier 3: Active OTA
            if (activeModelFile.exists()) {
                mapFileToReadOnlyBuffer(activeModelFile)?.let {
                    activeModelBuffer = it
                    if (activeManifestFile.exists()) {
                        currentHeuristics = parseHeuristics(activeManifestFile.readText())
                    }
                    Log.i(TAG, "Loaded Tier-3 (Active OTA) model.")
                    return@write
                }
            }

            // Tier 2: Rollback OTA
            if (rollbackModelFile.exists()) {
                mapFileToReadOnlyBuffer(rollbackModelFile)?.let {
                    activeModelBuffer = it
                    Log.w(TAG, "Loaded Tier-2 (Rollback) model.")
                    return@write
                }
            }

            // Tier 1: Factory Asset Fallback
            try {
                val manifestStr = context.assets.open("manifest.json").bufferedReader().use { it.readText() }
                currentHeuristics = parseHeuristics(manifestStr)
                val modelName = JSONObject(manifestStr).optString("model_file", "yolov8n_int8_factory.tflite")
                context.assets.openFd(modelName).use { afd ->
                    FileInputStream(afd.fileDescriptor).channel.use { channel ->
                        activeModelBuffer = channel.map(
                            FileChannel.MapMode.READ_ONLY,
                            afd.startOffset,
                            afd.declaredLength
                        )
                    }
                }
                Log.i(TAG, "Loaded Tier-1 (Factory Asset) fallback model.")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load Tier-1 factory fallback assets: ${e.message}")
            }
        }
    }

    private fun mapFileToReadOnlyBuffer(file: File): ByteBuffer? {
        return try {
            FileInputStream(file).channel.use { channel ->
                channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size())
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to memory-map file ${file.name}: ${e.message}")
            null
        }
    }

    private fun parseHeuristics(manifestJson: String): Heuristics {
        return try {
            val root = JSONObject(manifestJson)
            val h = root.optJSONObject("heuristics") ?: return Heuristics()
            Heuristics(
                confidenceThreshold = h.optDouble("confidence_threshold", 0.45).toFloat(),
                iouThreshold = h.optDouble("iou_threshold", 0.50).toFloat(),
                cooldownMs = h.optLong("cooldown_ms", 3500L),
                criticalAreaRatio = h.optDouble("critical_area_ratio", 0.18).toFloat()
            )
        } catch (e: Exception) {
            Heuristics()
        }
    }

    private fun computeSha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { fis ->
            val buffer = ByteArray(65536)
            var read = fis.read(buffer)
            while (read != -1) {
                digest.update(buffer, 0, read)
                read = fis.read(buffer)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    companion object {
        private const val TAG = "AdaptiveModelManager"
    }
}
