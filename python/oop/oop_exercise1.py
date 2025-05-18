class Estudiante():
    def __init__(self, nombre, edad, grado):
        self.nombre = nombre
        self.edad = edad
        self.grado = grado
    
    def estudiar(self):
        print(f"el estudiante {self.nombre} está estudiando")

# Add variables
nombre = input("Ingrese el nombre del estudiante: ")
edad = int(input("Ingrese la edad del estudiante: "))
grado = input("Ingrese el grado del estudiante: ")

estudiante = Estudiante(nombre, edad, grado)

estudiar_command = input("Indique 'estudiar' para que el estudiante estudie: ")
if estudiar_command == "estudiar":
    estudiante.estudiar()
