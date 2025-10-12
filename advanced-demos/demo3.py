import asyncio
import aiohttp
import time

# Reemplaza con tu clave API real
API_KEY = "tu_clave_api_aquí"
API_URL = "https://api.openai.com/v1/chat/completions"


async def generar_texto(prompt, modelo="gpt-3.5-turbo"):
    """Genera texto usando una API de IA de manera asíncrona"""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}

    datos = {
        "model": modelo,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
    }

    # Simulamos la respuesta de la API para la demostración
    # En un caso real, usaríamos:
    async with aiohttp.ClientSession() as session:
        async with session.post(API_URL, headers=headers, json=datos) as response:
            if response.status == 200:
                resultado = await response.json()
                return resultado["choices"][0]["message"]["content"].strip()

    # Para la demo, simplemente simulamos el tiempo de respuesta
    #await asyncio.sleep(2)  # Simula el tiempo de respuesta de la API

    # Respuestas simuladas
    respuestas = {
        "Explica la programación asíncrona.": "La programación asíncrona permite ejecutar tareas sin bloquear el hilo principal, mejorando la eficiencia en operaciones de E/S.",
        "¿Qué es Python?": "Python es un lenguaje de programación interpretado, de alto nivel y propósito general conocido por su legibilidad y versatilidad.",
        "Dame tres consejos de productividad.": "1. Prioriza tareas importantes. 2. Usa la técnica Pomodoro. 3. Automatiza tareas repetitivas.",
    }

    return respuestas.get(
        prompt, "No tengo una respuesta predefinida para esta pregunta."
    )


async def main():
    # Lista de prompts para procesar concurrentemente
    prompts = [
        "Explica la programación asíncrona.",
        "¿Qué es Python?",
        "Dame tres consejos de productividad.",
    ]

    print("Iniciando llamadas API concurrentes...")
    inicio = time.time()

    # Procesar todos los prompts concurrentemente
    resultados = await asyncio.gather(*(generar_texto(prompt) for prompt in prompts))

    fin = time.time()
    print(f"Todas las tareas completadas en {fin - inicio:.2f} segundos\n")

    # Mostrar resultados
    for i, resultado in enumerate(resultados, 1):
        print(f"Resultado {i}:\n{resultado}\n")

    # Para comparación, mostrar cuánto tiempo tomaría el procesamiento secuencial
    print("Ahora procesando secuencialmente para comparación...")
    inicio = time.time()

    for prompt in prompts:
        await generar_texto(prompt)

    fin = time.time()
    print(f"El procesamiento secuencial tomaría {fin - inicio:.2f} segundos")


# Para ejecutar en un script normal:
if __name__ == "__main__":
    asyncio.run(main())
