# Manual

## Helios++

### Via script

Ejecute `ubuntu_dependencies.sh` y `helios_linux_install.sh` dentro del directorio instalacion, en ese orden. Esto instalará Helios++ en su máquina. Este script asume que su distribución es Ubuntu.

El directorio de instalación de Helios++ es `$HOME/git/helios`

### Vía docker 

Ejecute `build_exec_docker_image.sh`. Esto creará una imagen basada en Ubuntu con Helios++ ya instalado y creará el contenedor con los contenidos necesarios del workshop. Es necesario tener docker instalado en su máquina.

El directorio de instalación será $HOME/git/helios

### QoL

Añada el directorio de Helios++ a su `PATH` para poder ejecutar las simulaciones de Helios++ desde cualquier lugar. Para ello, añada la siguiente línea a su archivo `.bashrc`:

```bash
export PATH=$PATH:$HOME/git/helios
```

## VirtuaLearn3D

### Vía script

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