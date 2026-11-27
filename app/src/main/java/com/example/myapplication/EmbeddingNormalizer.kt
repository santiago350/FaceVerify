package com.example.myapplication

import kotlin.math.sqrt

object EmbeddingNormalizer {

    private const val EMBEDDING_SIZE = 128
    private const val EPSILON = 1e-12

    fun normalize(embedding: FloatArray): FloatArray {

        require(embedding.size == EMBEDDING_SIZE) {
            "El embedding debe contener exactamente 128 valores."
        }

        var sumSquares = 0.0

        for (value in embedding) {
            require(value.isFinite()) {
                "El embedding contiene valores no válidos."
            }

            val number = value.toDouble()
            sumSquares += number * number
        }

        val norm = sqrt(sumSquares)

        require(norm.isFinite() && norm > EPSILON) {
            "No se puede normalizar un embedding vacío o inválido."
        }

        return FloatArray(embedding.size) { index ->
            (embedding[index].toDouble() / norm).toFloat()
        }
    }
}
