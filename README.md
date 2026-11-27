# FaceVerify

**Sistema Android de verificación facial mediante reconocimiento biométrico y prueba de vida.**

## Descripción

FaceVerify es una aplicación móvil desarrollada para Android que permite realizar procesos de registro, autenticación y verificación de identidad mediante reconocimiento facial.

El sistema utiliza modelos de aprendizaje automático para obtener características faciales y compararlas con las plantillas registradas por cada usuario. También incorpora pruebas de vida para comprobar que la persona frente a la cámara realice determinadas acciones.

## Funcionalidades principales

- Registro de usuarios.
- Captura y registro de plantillas faciales desde diferentes ángulos.
- Inicio de sesión mediante credenciales.
- Verificación facial mediante comparación de similitud coseno.
- Prueba de vida con desafíos aleatorios, como parpadear y mover la cabeza.
- Validación de condiciones de iluminación.
- Almacenamiento local de información y plantillas faciales protegidas.
- Resultado de verificación aceptado o rechazado.

## Tecnologías utilizadas

- **Plataforma:** Android.
- **Lenguaje:** Kotlin.
- **Entorno de desarrollo:** Android Studio.
- **Cámara:** CameraX.
- **Detección facial:** MediaPipe y ML Kit.
- **Reconocimiento facial:** MobileFaceNet.
- **Modelos:** TensorFlow Lite.
- **Base de datos:** SQLite.

## Estructura del proyecto

| Clase | Responsabilidad |
|---|---|
| `WelcomeActivity` | Pantalla de bienvenida. |
| `RegisterActivity` | Registro de usuarios. |
| `LoginActivity` | Autenticación mediante credenciales. |
| `ConsentActivity` | Consentimiento para el proceso biométrico. |
| `FaceEnrollmentActivity` | Registro de plantillas faciales. |
| `MainActivity` | Proceso de verificación facial y prueba de vida. |
| `FaceEmbeddingProcessor` | Procesamiento de características faciales. |
| `MediaPipeFaceDetector` | Detección facial mediante MediaPipe. |
| `BlinkLivenessDetector` | Detección de parpadeo. |
| `UserDatabase` | Gestión del almacenamiento local. |
| `UserMenuActivity` | Menú principal del usuario. |

## Modelos de reconocimiento

Los modelos utilizados se encuentran en `app/src/main/assets/`:

- `MediaPipeFaceDetector.tflite`
- `MediaPipeFaceLandmarkDetector.tflite`
- `mobile_facenet.tflite`

## Requisitos

- Android Studio.
- Android SDK compatible con la configuración del proyecto.
- Dispositivo Android con cámara frontal.
- Gradle, incluido mediante el Gradle Wrapper del proyecto.

## Ejecución

1. Clonar el repositorio:

   ```bash
   git clone https://github.com/santiago350/FaceVerify.git
   ```

2. Abrir el proyecto desde Android Studio.
3. Esperar la sincronización de Gradle.
4. Conectar un dispositivo Android o iniciar un emulador compatible.
5. Ejecutar la aplicación.

## Repositorio

[GitHub - FaceVerify](https://github.com/santiago350/FaceVerify)

---

**Proyecto académico desarrollado en Kotlin para la plataforma Android.**