
package com.example.myapplication

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class FaceSimilarityTest {

    @Test
    fun embeddingsIdenticosDebenTenerSimilitudUno() {
        val first = FloatArray(128) { 1f }
        val second = FloatArray(128) { 1f }

        val result = FaceSimilarity.cosineSimilarity(first, second)

        assertEquals(1.0, result, 0.000001)
    }

    @Test
    fun embeddingsOpuestosDebenTenerSimilitudMenosUno() {
        val first = FloatArray(128) { 1f }
        val second = FloatArray(128) { -1f }

        val result = FaceSimilarity.cosineSimilarity(first, second)

        assertEquals(-1.0, result, 0.000001)
    }

    @Test
    fun embeddingsOrtogonalesDebenTenerSimilitudCero() {
        val first = FloatArray(128)
        val second = FloatArray(128)

        first[0] = 1f
        second[1] = 1f

        val result = FaceSimilarity.cosineSimilarity(first, second)

        assertEquals(0.0, result, 0.000001)
    }

    @Test
    fun debeSerIndependienteDeLaMagnitudDeLosVectores() {
        val first = FloatArray(128) { 2f }
        val second = FloatArray(128) { 5f }

        val result = FaceSimilarity.cosineSimilarity(first, second)

        assertEquals(1.0, result, 0.000001)
    }

    @Test
    fun debeRechazarEmbeddingsConDimensionesIncorrectas() {
        val first = FloatArray(64)
        val second = FloatArray(128)

        assertThrows(IllegalArgumentException::class.java) {
            FaceSimilarity.cosineSimilarity(first, second)
        }
    }

    @Test
    fun debeRechazarVectoresDeCeros() {
        val first = FloatArray(128)
        val second = FloatArray(128) { 1f }

        assertThrows(IllegalArgumentException::class.java) {
            FaceSimilarity.cosineSimilarity(first, second)
        }
    }

    @Test
    fun debeRechazarValoresNoValidos() {
        val first = FloatArray(128) { 1f }
        val second = FloatArray(128) { 1f }

        first[10] = Float.NaN

        assertThrows(IllegalArgumentException::class.java) {
            FaceSimilarity.cosineSimilarity(first, second)
        }
    }
}
se