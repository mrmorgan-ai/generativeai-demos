import asyncio
import time


def tarea():
    print("Iniciando tarea...")
    time.sleep(3)  # No bloquea la ejecución, permite otras tareas
    print("Tarea completada.")


def tarea2():
    print("Probando tarea 2...")
    print("Tarea 2 completada.")

def main():
    print("Inicio del programa")
    tarea()
    tarea2()
    # await tarea()
    # await tarea2()
    print("Fin del programa")


if __name__ == "__main__":
    main()# Ejecuta el event loop
