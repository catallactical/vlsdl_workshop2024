# Manual

## Via script

Ejecute `ubuntu_dependencies.sh` y `helios_linux_install.sh` dentro del directorio instalacion, en ese orden. Esto instalará Helios++ en su máquina. Este script asume que su distribución es Ubuntu.

El directorio de instalación de Helios++ es `$HOME/git/helios`

## Vía docker 

Ejecute `build_exec_docker_image.sh`, esto creará una imagen basada en Ubuntu con Helios++ ya instalado y creará el contenedor con los contenidos necesarios del workshop. Es necesario tener docker instalado en su máquina.

El directorio de instalación será $HOME/git/helios

## QoL

Añada el directorio de Helios++ a su `PATH` para poder ejecutar las simulaciones de Helios++ desde cualquier lugar. Para ello, añada la siguiente línea a su archivo `.bashrc`:

```bash
export PATH=$PATH:$HOME/git/helios
```