package com.example.myapplication

import org.junit.Assert.assertEquals
import org.junit.Test

class BlinkLivenessDetectorTest {

    @Test
    fun estadoInicialDebeSerWaitingForOpen() {
        val detector = BlinkLivenessDetector()

        assertEquals(
            BlinkLivenessDetector.Status.WAITING_FOR_OPEN,
            detector.getStatus()
        )
    }

    @Test
    fun ojosAbiertosDebenAvanzarAlCierre() {
        val detector = BlinkLivenessDetector()

        val result = detector.update(0.90f, 0.90f)

        assertEquals(
            BlinkLivenessDetector.Status.WAITING_FOR_CLOSE,
            result
        )
    }

    @Test
    fun ojosCerradosDebenAvanzarALaReapertura() {
        val detector = BlinkLivenessDetector()

        detector.update(0.90f, 0.90f)
        val result = detector.update(0.10f, 0.10f)

        assertEquals(
            BlinkLivenessDetector.Status.WAITING_FOR_REOPEN,
            result
        )
    }

    @Test
    fun parpadeoCompletoDebeCompletarse() {
        val detector = BlinkLivenessDetector()

        detector.update(0.90f, 0.90f)
        detector.update(0.10f, 0.10f)
        val result = detector.update(0.90f, 0.90f)

        assertEquals(
            BlinkLivenessDetector.Status.COMPLETED,
            result
        )
    }

    @Test
    fun probabilidadesNulasNoDebenAvanzarElEstado() {
        val detector = BlinkLivenessDetector()

        val result = detector.update(null, 0.90f)

        assertEquals(
            BlinkLivenessDetector.Status.WAITING_FOR_OPEN,
            result
        )
    }

    @Test
    fun resetDebeRestablecerElEstadoInicial() {
        val detector = BlinkLivenessDetector()

        detector.update(0.90f, 0.90f)
        detector.reset()

        assertEquals(
            BlinkLivenessDetector.Status.WAITING_FOR_OPEN,
            detector.getStatus()
        )
    }

    @Test
    fun unSoloOjoCerradoNoDebeCompletarElParpadeo() {
        val detector = BlinkLivenessDetector()

        detector.update(0.90f, 0.90f)
        val result = detector.update(0.10f, 0.90f)

        assertEquals(
            BlinkLivenessDetector.Status.WAITING_FOR_CLOSE,
            result
        )
    }
}
