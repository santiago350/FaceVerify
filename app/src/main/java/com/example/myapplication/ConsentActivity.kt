
package com.example.myapplication

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import android.widget.CheckBox
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity

class ConsentActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContentView(R.layout.activity_consent)

        val checkConsent = findViewById<CheckBox>(R.id.checkConsent)
        val btnAccept = findViewById<Button>(R.id.btnAcceptConsent)
        val btnReject = findViewById<Button>(R.id.btnRejectConsent)

        // Inicialmente no se puede aceptar sin marcar la casilla
        btnAccept.isEnabled = false

        checkConsent.setOnCheckedChangeListener { _, isChecked ->
            btnAccept.isEnabled = isChecked
            btnAccept.alpha = if (isChecked) 1f else 0.5f
        }

        btnAccept.setOnClickListener {

            if (!checkConsent.isChecked) {
                Toast.makeText(
                    this,
                    "Debes autorizar el tratamiento de datos para continuar.",
                    Toast.LENGTH_SHORT
                ).show()
                return@setOnClickListener
            }

            // Autorizó: continuar al registro
            startActivity(Intent(this, RegisterActivity::class.java))
            finish()
        }

        btnReject.setOnClickListener {
            // Rechazó: no se registra la cuenta y se vuelve al inicio
            finish()
        }
    }
}
