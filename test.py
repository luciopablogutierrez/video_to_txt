from pyannote.audio import Inference
from pydub import AudioSegment
import speech_recognition as sr
import os
from huggingface_hub import login

# Paso 1: Extraer el audio del archivo MP4
def extraer_audio(mp4_path, wav_path):
    """
    Extrae el audio de un archivo MP4 y lo guarda como un archivo WAV.
    :param mp4_path: Ruta del archivo MP4.
    :param wav_path: Ruta donde se guardará el archivo WAV.
    """
    try:
        print("Extrayendo audio del archivo MP4...")
        audio = AudioSegment.from_file(mp4_path, format="mp4")
        audio.export(wav_path, format="wav")
        print(f"Audio extraído y guardado en {wav_path}")
    except Exception as e:
        print(f"Error al extraer el audio: {e}")
        raise

# Paso 2: Segmentación de hablantes
def segmentar_hablantes(wav_path):
    """
    Segmenta el audio en función de los hablantes, utilizando el modelo de Hugging Face.
    :param wav_path: Ruta del archivo WAV.
    :return: Lista de segmentos con los hablantes identificados.
    """
    try:
        print("Segmentando hablantes...")
        
        # Cargar el modelo de segmentación
        inference = Inference("philschmid/pyannote-segmentation")
        
        # Aplicar la segmentación al archivo de audio
        segmentation = inference(wav_path)
        
        # Convertir la segmentación a una lista de segmentos
        segmentos = []
        for segment in segmentation:
            start = segment[0].start
            end = segment[0].end
            speaker = segment[1]
            segmentos.append({
                "start": start,
                "end": end,
                "speaker": speaker
            })
        
        print("Segmentación completada.")
        return segmentos
    
    except Exception as e:
        print(f"Error durante la segmentación: {e}")
        return None

# Paso 3: Transcribir el audio a texto
def transcribir_audio(wav_path, segmentos, lenguaje="es-ES"):
    """
    Transcribe un archivo de audio WAV a texto, diferenciando entre hablantes.
    :param wav_path: Ruta del archivo WAV.
    :param segmentos: Lista de segmentos con los hablantes identificados.
    :param lenguaje: Idioma del audio (por defecto es español de España).
    :return: Texto transcrito con identificadores de hablantes.
    """
    recognizer = sr.Recognizer()
    texto_transcrito = ""

    print("Transcribiendo audio a texto...")
    for segmento in segmentos:
        start = int(segmento["start"] * 1000)  # Convertir a milisegundos
        end = int(segmento["end"] * 1000)      # Convertir a milisegundos
        speaker = segmento["speaker"]

        try:
            # Extraer el segmento de audio
            audio = AudioSegment.from_wav(wav_path)
            segmento_audio = audio[start:end]

            # Guardar el segmento temporalmente
            segmento_path = "temp_segmento.wav"
            segmento_audio.export(segmento_path, format="wav")

            # Transcribir el segmento
            with sr.AudioFile(segmento_path) as source:
                audio_segment = recognizer.record(source)

            try:
                texto = recognizer.recognize_google(audio_segment, language=lenguaje)
                texto_transcrito += f"{speaker}: {texto}\n"
            except sr.UnknownValueError:
                texto_transcrito += f"{speaker}: [Ininteligible]\n"
            except sr.RequestError as e:
                texto_transcrito += f"{speaker}: [Error en la transcripción]\n"
        
        except Exception as e:
            print(f"Error al procesar el segmento {speaker} ({start}-{end}): {e}")
            texto_transcrito += f"{speaker}: [Error al procesar el segmento]\n"
        
        finally:
            # Eliminar el archivo temporal
            if os.path.exists(segmento_path):
                os.remove(segmento_path)

    print("Transcripción completada.")
    return texto_transcrito

# Paso 4: Guardar el texto en un archivo
def guardar_texto(texto, archivo_salida):
    """
    Guarda el texto transcrito en un archivo.
    :param texto: Texto a guardar.
    :param archivo_salida: Ruta del archivo de salida.
    """
    try:
        with open(archivo_salida, "w", encoding="utf-8") as file:
            file.write(texto)
        print(f"Texto guardado en {archivo_salida}")
    except Exception as e:
        print(f"Error al guardar el texto: {e}")

# Paso 5: Función principal
def main():
    # Rutas de los archivos
    mp4_path = "test.mp4"  # Archivo MP4 de entrada
    wav_path = "test.wav"  # Archivo WAV temporal
    archivo_salida = "transcripcion.txt"  # Archivo de salida para el texto transcrito

    # Iniciar sesión en Hugging Face
    try:
        login()  # Esto abrirá una ventana de autenticación en el navegador
    except Exception as e:
        print(f"Error al iniciar sesión en Hugging Face: {e}")
        return

    try:
        # Extraer el audio del MP4
        extraer_audio(mp4_path, wav_path)

        # Segmentación de hablantes
        segmentos = segmentar_hablantes(wav_path)
        
        if segmentos is None:
            print("No se pudo realizar la segmentación. Verifica tu sesión.")
            return

        # Transcribir el audio a texto
        texto = transcribir_audio(wav_path, segmentos)

        # Guardar el texto en un archivo
        if texto:  # Solo guardar si se obtuvo una transcripción
            guardar_texto(texto, archivo_salida)
            print("Texto transcrito:")
            print(texto)
        else:
            print("No se pudo transcribir el audio.")

    except Exception as e:
        print(f"Error en la ejecución del programa: {e}")

# Ejecutar el programa
if __name__ == "__main__":
    main()
