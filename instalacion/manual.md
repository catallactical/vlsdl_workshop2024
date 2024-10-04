# Manual

## Índice

1. [Helios++](#helios)
1.1. [Configuración del script](#configuración-del-script)
1.2. [Instalación](#instalación)
2. [VirtuaLearn3D](#virtualearn3d)
2.1. [Compilación de _bindings_](#compilación-de-bindings)

## Helios++

### Configuración del script

Primero, compruebe la versión de Python instalada en su sistema:
  
  ```bash
  python3 --version
  ```
Quédese con la notación MAJOR.MINOR. Por ejemplo, si la versión es 3.8.5, quédese con 3.8.

### Instalación

#### Vía script

En `helios_linux_install.sh`, sustituya `PYTHON_DOT_VERSION` por la notación MAJOR.MINOR que obtuvo en el paso anterior.

Ejecute `ubuntu_dependencies.sh` y `helios_linux_install.sh` dentro del directorio instalacion, en ese orden. Esto instalará Helios++ en su máquina. Este script asume que su distribución es Ubuntu.

El directorio de instalación de Helios++ es `$HOME/git/helios`

#### Vía docker 

Ejecute `build_exec_docker_image.sh`. Esto creará una imagen basada en Ubuntu con Helios++ ya instalado y creará el contenedor con los contenidos necesarios del workshop. Es necesario tener docker instalado en su máquina.

El directorio de instalación será $HOME/git/helios

### QoL

Añada el directorio de Helios++ a su `PATH` para poder ejecutar las simulaciones de Helios++ desde cualquier lugar. Para ello, ejecute en su terminal o añada la siguiente línea a su archivo `.bashrc`:

```bash
export PATH=$PATH:$HOME/git/helios
```

## VirtuaLearn3D

### Compilación de _bindings_

El proyecto ya se encuentra en este mismo directorio. Se deben compilar los _bindings_ de C++. Para ello, ejecute `vl3d_linux_install.sh` desde el directorio de instalación. El _script_ instala un entorno conda llamado **vl3d**. Cuando termine la ejecución del _script_, debe configurar conda y activar el entorno **vl3d**. Para ello, ejecute:

```bash
export PATH=$HOME/miniconda3/bin:$PATH
conda init
source $HOME/.bashrc # NOTA: Si su archivo de configuración es diferente, cámbielo por el correcto.
conda activate vl3d
```

### Ejecución de _pipelines_

Los pipelines están configurados en archivos _json_. Para ejecutar un pipeline, ejecute:

```bash
python vl3d.py --pipeline <ruta_al_pipeline.json>
```