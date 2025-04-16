import asyncio
# import time


async def tarea():
    print("Iniciando tarea...")
    await asyncio.sleep(3)  # No bloquea la ejecución, permite otras tareas
    print("Tarea completada.")


async def tarea2():
    print("Probando tarea 2...")
    print("Tarea 2 completada.")

async def main():
    print("Inicio del programa")
    await asyncio.gather(tarea(), tarea2())
    # await tarea()
    # await tarea2()
    print("Fin del programa")


if __name__ == "__main__":
    asyncio.run(main()) # Ejecuta el event loop
