package com.example.myapplication

import kotlin.math.sqrt

object FaceSimilarity {

    fun cosineSimilarity(
        first: FloatArray,
        second: FloatArray
    ): Double {

        require(
            first.size == 128 &&
                    second.size == 128
        ) {
            "Los embeddings deben tener exactamente 128 dimensiones."
        }

        var dotProduct = 0.0
        var normFirst = 0.0
        var normSecond = 0.0

        for (i in first.indices) {
            require(
                first[i].isFinite() &&
                        second[i].isFinite()
            ) {
                "Embedding contiene valores inválidos."
            }

            val a = first[i].toDouble()
            val b = second[i].toDouble()

            dotProduct += a * b
            normFirst += a * a
            normSecond += b * b
        }

        val denominator =
            sqrt(normFirst) * sqrt(normSecond)

        require(
            denominator > 1e-12 &&
                    denominator.isFinite()
        ) {
            "No se pudo calcular la similitud."
        }

        val similarity = dotProduct / denominator

        require(similarity.isFinite()) {
            "El resultado de similitud no es válido."
        }

        return similarity.coerceIn(-1.0, 1.0)
    }
}
