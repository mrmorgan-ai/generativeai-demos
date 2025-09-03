import asyncio
import time


async def tarea(nombre, segundos):
    """Simula una tarea asíncrona que toma cierto tiempo"""
    print(f"Iniciando tarea '{nombre}'")
    await asyncio.sleep(segundos)  # Simula una operación de E/S
    print(f"Tarea '{nombre}' completada después de {segundos} segundos")
    return f"Resultado de {nombre}"


async def main():
    print("=== Ejecución Secuencial ===")
    inicio = time.time()

    # Ejecución secuencial
    for i in range(1, 4):
        await tarea(f"Secuencial-{i}", i)

    fin = time.time()
    print(f"Tiempo total (secuencial): {fin - inicio:.2f} segundos\n")

    print("=== Ejecución Concurrente ===")
    inicio = time.time()

    # Ejecución concurrente con gather
    resultados = await asyncio.gather(
        tarea("Concurrente-1", 1), tarea("Concurrente-2", 2), tarea("Concurrente-3", 3)
    )

    fin = time.time()
    print(f"Tiempo total (concurrente): {fin - inicio:.2f} segundos")
    print(f"Resultados: {resultados}")


# Para ejecutar en un script normal:
if __name__ == "__main__":
    asyncio.run(main())
