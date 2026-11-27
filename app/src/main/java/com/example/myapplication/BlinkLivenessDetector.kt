
package com.example.myapplication

class BlinkLivenessDetector {

    enum class Status {
        WAITING_FOR_OPEN,
        WAITING_FOR_CLOSE,
        WAITING_FOR_REOPEN,
        COMPLETED
    }

    companion object {
        // Umbrales más tolerantes con las variaciones de ML Kit
        private const val CLOSED_THRESHOLD = 0.48f
        private const val OPEN_THRESHOLD = 0.52f

        // Menos frames necesarios
        private const val REQUIRED_OPEN_FRAMES = 1
        private const val REQUIRED_CLOSED_FRAMES = 1
        private const val REQUIRED_REOPEN_FRAMES = 1

        // Evita que un solo ojo dé una señal extrema
        private const val MAX_EYE_DIFFERENCE = 0.40f
    }

    private var status = Status.WAITING_FOR_OPEN

    private var openFrames = 0
    private var closedFrames = 0
    private var reopenFrames = 0

    fun update(
        leftEyeProbability: Float?,
        rightEyeProbability: Float?
    ): Status {

        if (status == Status.COMPLETED) return status

        if (
            leftEyeProbability == null ||
            rightEyeProbability == null ||
            !leftEyeProbability.isFinite() ||
            !rightEyeProbability.isFinite()
        ) {
            resetCurrentFrames()
            return status
        }

        val left = leftEyeProbability.coerceIn(0f, 1f)
        val right = rightEyeProbability.coerceIn(0f, 1f)

        val average = (left + right) / 2f
        val eyeDifference = kotlin.math.abs(left - right)

        val bothOpen =
            left >= OPEN_THRESHOLD &&
                    right >= OPEN_THRESHOLD

        val bothClosed =
            average <= CLOSED_THRESHOLD &&
                    eyeDifference <= MAX_EYE_DIFFERENCE

        when (status) {

            Status.WAITING_FOR_OPEN -> {
                if (bothOpen) {
                    openFrames++
                } else {
                    openFrames = 0
                }

                if (openFrames >= REQUIRED_OPEN_FRAMES) {
                    status = Status.WAITING_FOR_CLOSE
                    closedFrames = 0
                }
            }

            Status.WAITING_FOR_CLOSE -> {
                if (bothClosed) {
                    closedFrames++
                } else {
                    closedFrames = 0
                }

                if (closedFrames >= REQUIRED_CLOSED_FRAMES) {
                    status = Status.WAITING_FOR_REOPEN
                    reopenFrames = 0
                }
            }

            Status.WAITING_FOR_REOPEN -> {
                if (bothOpen) {
                    reopenFrames++
                } else {
                    reopenFrames = 0
                }

                if (reopenFrames >= REQUIRED_REOPEN_FRAMES) {
                    status = Status.COMPLETED
                }
            }

            Status.COMPLETED -> Unit
        }

        return status
    }

    private fun resetCurrentFrames() {
        when (status) {
            Status.WAITING_FOR_OPEN -> openFrames = 0
            Status.WAITING_FOR_CLOSE -> closedFrames = 0
            Status.WAITING_FOR_REOPEN -> reopenFrames = 0
            Status.COMPLETED -> Unit
        }
    }

    fun reset() {
        status = Status.WAITING_FOR_OPEN
        openFrames = 0
        closedFrames = 0
        reopenFrames = 0
    }

    fun getStatus(): Status = status
}
