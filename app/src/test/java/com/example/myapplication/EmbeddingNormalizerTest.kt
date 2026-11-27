package com.example.myapplication

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test
import kotlin.math.sqrt

class EmbeddingNormalizerTest {

    @Test
    fun debeNormalizarUnVectorDe128Valores() {
        val input = FloatArray(128) { 1f }

        val result = EmbeddingNormalizer.normalize(input)

        assertEquals(128, result.size)

        val norm = sqrt(result.sumOf {
            it.toDouble() * it.toDouble()
        })

        assertEquals(1.0, norm, 0.000001)
    }

    @Test
    fun debeConservarLaDireccionDelVector() {
        val input = FloatArray(128) { index ->
            if (index == 0) 3f else 0f
        }

        val result = EmbeddingNormalizer.normalize(input)

        assertEquals(1f, result[0], 0.000001f)
        assertEquals(0f, result[1], 0.000001f)
    }

    @Test
    fun debeRechazarUnVectorConTamanoIncorrecto() {
        val input = FloatArray(64) { 1f }

        assertThrows(IllegalArgumentException::class.java) {
            EmbeddingNormalizer.normalize(input)
        }
    }

    @Test
    fun debeRechazarUnVectorDeCeros() {
        val input = FloatArray(128)

        assertThrows(IllegalArgumentException::class.java) {
            EmbeddingNormalizer.normalize(input)
        }
    }

    @Test
    fun debeRechazarValoresNoFinitos() {
        val input = FloatArray(128) { 1f }
        input[10] = Float.NaN

        assertThrows(IllegalArgumentException::class.java) {
            EmbeddingNormalizer.normalize(input)
        }
    }
}
