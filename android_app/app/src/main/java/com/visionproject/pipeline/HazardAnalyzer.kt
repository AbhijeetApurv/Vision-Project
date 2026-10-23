package com.visionproject.pipeline

import android.content.Context
import android.os.Build
import android.os.SystemClock
import android.os.VibrationEffect
import android.os.Vibrator
import android.speech.tts.TextToSpeech
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.visionproject.ml.AdaptiveModelManager
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap

/**
 * PLANE 3: CameraX 30 FPS real-time frame analyzer coupled with
 * per-class anti-fatigue audio/haptic cooldowns.
 */
class HazardAnalyzer(
    context: Context,
    private val modelManager: AdaptiveModelManager,
    private val onDetectionsListener: (List<Detection>) -> Unit = {}
) : ImageAnalysis.Analyzer {

    data class Detection(
        val label: String,
        val confidence: Float,
        val normalizedArea: Float,
        val centerXRatio: Float
    )

    private val lastAlertTimestampByClass = ConcurrentHashMap<String, Long>()
    private val vibrator = context.getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
    private var tts: TextToSpeech? = null

    init {
        tts = TextToSpeech(context.applicationContext) { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts?.language = Locale.US
            }
        }
    }

    override fun analyze(image: ImageProxy) {
        try {
            val detections = modelManager.withModelBuffer { buffer, heuristics ->
                runInference(image, buffer, heuristics)
            }.orEmpty()

            if (detections.isNotEmpty()) {
                onDetectionsListener(detections)
                evaluateAndTriggerFeedback(detections, modelManager.currentHeuristics)
            }
        } finally {
            image.close()
        }
    }

    private fun runInference(
        image: ImageProxy,
        modelBuffer: java.nio.ByteBuffer,
        heuristics: AdaptiveModelManager.Heuristics
    ): List<Detection> {
        // Placeholder for LiteRT Interpreter invocation on YUV_420_888 / RGB tensor
        // Filters candidate boxes by heuristics.confidenceThreshold & heuristics.iouThreshold
        return emptyList()
    }

    private fun evaluateAndTriggerFeedback(
        detections: List<Detection>,
        heuristics: AdaptiveModelManager.Heuristics
    ) {
        val now = SystemClock.elapsedRealtime()
        val highestPriority = detections
            .filter { it.confidence >= heuristics.confidenceThreshold }
            .maxByOrNull { it.normalizedArea } ?: return

        val lastAlert = lastAlertTimestampByClass[highestPriority.label] ?: 0L
        if (now - lastAlert < heuristics.cooldownMs) {
            return // Suppress alert to prevent sensory fatigue
        }

        lastAlertTimestampByClass[highestPriority.label] = now
        val direction = when {
            highestPriority.centerXRatio < 0.35f -> "on your left"
            highestPriority.centerXRatio > 0.65f -> "on your right"
            else -> "ahead"
        }

        val isCritical = highestPriority.normalizedArea >= heuristics.criticalAreaRatio
        triggerHaptics(isCritical)
        tts?.speak(
            "${highestPriority.label} $direction",
            if (isCritical) TextToSpeech.QUEUE_FLUSH else TextToSpeech.QUEUE_ADD,
            null,
            "hazard_${highestPriority.label}_$now"
        )
    }

    private fun triggerHaptics(isCritical: Boolean) {
        val vib = vibrator ?: return
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val effect = if (isCritical) {
                VibrationEffect.createWaveform(longArrayOf(0, 180, 80, 220), -1)
            } else {
                VibrationEffect.createOneShot(90, VibrationEffect.DEFAULT_AMPLITUDE)
            }
            vib.vibrate(effect)
        }
    }

    fun shutdown() {
        tts?.stop()
        tts?.shutdown()
    }
}
