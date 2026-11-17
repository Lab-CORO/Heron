# Prompt : porter la génération Heron vers ROS 2 Jazzy + cuRobo v2 et construire son image

Tu travailles sur **Heron** (dépôt `CapaciNet`, nom temporaire), un pipeline qui génère des cartes
d'atteignabilité de robots en voxels, puis entraîne un U-Net 3D dessus. La génération est un paquet
ROS 2 C++, `data_generation/`. Il appelle les services de cuRobo exposés par le paquet `curobo_ros`
(dépôt `Lab-CORO/curobo_ros`).

Aujourd'hui, la génération tourne sur un cluster SLURM sans Docker (Alliance Canada, Narval, A100),
dans une image Apptainer construite à partir d'une image Docker : Ubuntu 22.04, ROS 2 Humble,
cuRobo v1. `curobo_ros` est passé à **ROS 2 Jazzy et cuRobo v0.8.0 (API v2)**, et son API ROS a
changé. Ta mission tient en deux phases, **A puis B**, dans la même session :

- **A** : porter `data_generation` vers le nouveau `curobo_ros`.
- **B** : construire l'image de génération, puis la convertir en `.sif` Apptainer.

Tu travailles sur un poste du labo qui a Docker, sudo, un GPU NVIDIA et Apptainer. Le cluster n'a
pas Docker : seul le `.sif` final y sera copié.

## Règles

- Ne pousse rien et n'ouvre aucune PR, ni sur `Lab-CORO/CapaciNet`, ni sur `Lab-CORO/curobo_ros`,
  ni sur `curobo_msgs`. Committe localement, sur une branche `feat/generation-jazzy` créée à partir
  de la branche que l'on t'a donnée.
- Les identifiants personnels (comptes SLURM comme `def-xxx_gpu`, chemins `/home/<user>`)
  n'entrent jamais dans git. Les réglages propres à une machine vont dans `site.yaml`, qui est
  ignoré par git (voir `site.example.yaml`).
- Avant de changer un comportement, lis `GLOSSARY.md` (vocabulaire : Robot, Station, Dataset,
  Corpus…), `docs/images.md` (contrats des environnements) et
  `data_generation/pipeline/README.md`.
- Chaque fois que tu épingles une version, vérifie-la (tag ou commit existant, image disponible).
  N'écris jamais de version de mémoire.
- Si un point ci-dessous se révèle faux en pratique, arrête-toi et explique-le, au lieu de
  contourner en silence.

## État de départ (vérifié le 2026-11-17)

**Image actuelle** (`docker_images/data_generation.sif`, construite par
`data_generation/docker/x86.dockerfile` à partir de `curobo_ros:ampere-dev`) :
- Ubuntu 22.04, Python 3.10, CUDA 12.3, torch 2.2 (NGC), ROS 2 Humble, cuRobo v1 ;
- HighFive v2.8.0 et cnpy compilés depuis les sources, et Open3D compilé au commit `1868f43` ;
- les paquets `curobo_ros`, `curobo_msgs`, `CapaciNet/data_generation` et `doosan-robot2` (Humble),
  construits dans `/home/ros2_ws`.

**Nouveau `curobo_ros`** (`Lab-CORO/curobo_ros`, main au commit
`06aef2065c7a38625e1705048e5769eaa99c6b7d`, 2026-08-09) :
- `docker/x86_jazzy_prod.dockerfile` : `FROM nvcr.io/nvidia/pytorch:24.11-py3` (Ubuntu 24.04,
  Python 3.12), cuRobo `v0.8.0` installé avec l'extra `[cu12,dev,usd]`, ROS 2 Jazzy desktop,
  `rmw_fastrtps_cpp`, workspace `/home/ros2_ws` ;
- `docker/build_docker.sh` : choix Ampere ⇒ `TORCH_CUDA_ARCH_LIST="8.0 8.6"` ;
- le nœud `curobo_ik` n'existe plus. Il est remplacé par un nœud unifié : exécutable
  `curobo_trajectory_planner`, nom de nœud `unified_planner`. Ses services s'appellent
  `/<nœud>/ik_batch`, `/<nœud>/fk`, `/<nœud>/get_voxel_grid`, etc. (voir
  `curobo_ros/core/ik_services.py` et `obstacle_manager.py`) ;
- `curobo_msgs` main (commit `39ceb49`) contient toujours `IkBatch`, `AddObject`,
  `RemoveObject`, `GenerateRM`, `SceneGenerator` et `GetCollisionDistance`. **`GetVoxelGrid`
  prend maintenant dans sa requête une boîte englobante** (`bbox_min_x/y/z`,
  `bbox_max_x/y/z`, en mètres, repère de la base).

**Ce que `data_generation` appelle aujourd'hui** (`data_generation/src/*.cpp`) :
`/curobo_ik/ik_batch_poses` (IkBatch), `/curobo_ik/add_object`, `/curobo_ik/remove_object`,
`/curobo_ik/remove_all_objects` (Trigger), `/curobo_ik/get_voxel_grid` (requête vide,
`data_generator.cpp` vers la ligne 224) et `/curobo_ik/get_collision_distance`, plus ses propres
services `/generate_rm` et `/generate_scene`. Les fichiers launch (`data_generation/launch/*.py`)
démarrent `executable='curobo_ik', name='curobo_ik'`.

**Patchs actuels** (`data_generation/pipeline/curobo_patch/`, montés par bind-mount sur l'ancienne
image) :
- `obstacle_manager.py` : étendue de la scène lue dans `HERON_SCENE_HALF_EXTENT`, au lieu des
  ±1,52 m codés en dur. Cette étendue vient de la Station (`stations/<nom>.yaml`,
  `scene_half_extent`) ;
- `ros_service_manager.py` : service `load_voxel_scene`, qui ne sert qu'au mode `scene`.

**Lancement sur le cluster** : `data_generation/pipeline/gen_dataset.sh` prépare les fichiers et
soumet `data_generation/pipeline/job.sbatch`, qui lance
`apptainer exec -C --net --network=none --nv <sif> bash run_generation.sh` avec ces montages :
- le dossier de la tâche sur `/home/ros2_ws/install/data_generation/share/data_generation/data` ;
- les launch files sur `…/share/data_generation/launch` ;
- `$SLURM_TMPDIR` sur `/tmp` ;
- un cache warp.

Les paramètres arrivent par des variables `APPTAINERENV_*` (`MODE`, `VOXEL`, `DATASET_SIZE`,
`BATCH_SIZE`, `INIT_BATCH_SIZE`, `OBJ_MAX`, `REACH_MAX`, `ROBOT_CONFIG_FILE`, `BASE_LINK`,
`WORLD_FILE`, `CAMERA_FRAME`, `SLURM_ARRAY_TASK_ID`, `HERON_SCENE_HALF_EXTENT`).
`run_generation.sh` exporte `RMW_IMPLEMENTATION=rmw_fastrtps_cpp` et `ROS_LOCALHOST_ONLY=1`.

**Usage réel** : robot Doosan m1013 (`robots/m1013.yaml`), station `leeloo`, voxel 0,02 m,
scènes de 152³ voxels. Une scène prend environ 1 h 30 sur A100 (4023 lots de 10 000 poses d'IK).

## Phase A : porter `data_generation`

Pour développer et tester, construis l'image de base `curobo_ros` Jazzy x86 prod au commit indiqué
plus haut (voir la phase B, étape 1). Travaille ensuite dans un conteneur de cette image, avec
CapaciNet monté dans `/home/ros2_ws/src/CapaciNet`, et `colcon build` dedans.

1. **Nom du nœud cuRobo en paramètre.**
   - Plus aucun `/curobo_ik/...` en dur dans le C++ : chaque nœud qui appelle cuRobo déclare un
     paramètre ROS `curobo_node` (par défaut `unified_planner`) et forme les noms de service
     `"/" + curobo_node + "/ik_batch"`, etc.
   - Fais correspondre chaque ancien service à son équivalent v2. Lis le code de
     `curobo_ros/core/` pour les noms exacts : par exemple, `ik_batch_poses` devient `ik_batch`.
   - Vérifie champ par champ que les requêtes et réponses de chaque `.srv` utilisées par le C++
     correspondent à `curobo_msgs` main, et adapte le C++.
2. **Étendue de la scène sans patch.**
   - `generate_data` reçoit un paramètre `scene_half_extent` (mètres) et remplit la boîte de
     `GetVoxelGrid` avec ±`scene_half_extent` sur x, y et z.
   - Les launch files lisent `HERON_SCENE_HALF_EXTENT` depuis l'environnement et le passent en
     paramètre. Sans cette variable, la valeur par défaut est 1,52.
   - Le résultat doit rester une grille de `2 × scene_half_extent / voxel` voxels de côté
     (152³ pour leeloo à 0,02). Vérifie l'arrondi et l'origine en lisant ce que `generate_data`
     faisait de la grille Humble.
3. **Launch files** (`data_generation/launch/*.py`).
   - Ils lancent `curobo_trajectory_planner` sous le nom `unified_planner`, avec les paramètres
     qu'il attend : configuration du robot, `voxel_size`, `base_link`, monde… Relis
     `unified_planner_node.py` et `config_wrapper.py` pour leurs noms exacts en v2.
   - Ils passent `curobo_node` à nos nœuds.
4. **Open3D retiré.** Il n'apparaît que dans `data_generation/CMakeLists.txt` (`find_package` et
   les includes) et n'est utilisé dans aucun fichier C++. Supprime-le du CMake, après avoir vérifié
   avec `grep` qu'aucun code ne l'utilise.
5. **Variables ROS de Jazzy.** Dans `run_generation.sh`, remplace `ROS_LOCALHOST_ONLY=1`
   (obsolète en Jazzy) par `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`, et garde
   `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`.
6. **Cache warp.** Dans `job.sbatch`, la cible de montage `/home/willore/.cache/warp` devient
   `$HOME/.cache/warp`. Vérifie ce que vaut `HOME` dans un `apptainer exec -C`, et dans quel
   dossier warp écrit réellement.
7. **Mode `scene` : pas encore porté.** La nouvelle image n'a pas `load_voxel_scene`. Avec une image
   Jazzy, `gen_dataset.sh --mode scene` s'arrête avec ce message :
   « le mode scene n'est pas encore disponible avec l'image ROS 2 Jazzy + cuRobo v2 (fonction non
   portée) ; utilise l'ancienne image avec --sif <data_generation humble.sif> ».
   - La détection doit être fiable : lis un label de l'image (`apptainer inspect`, voir B.4), pas le
     nom du fichier.
   - Garde `data_generation/pipeline/curobo_patch/` et son montage, mais seulement pour l'ancienne
     image. Avec la nouvelle, `job.sbatch` ne monte aucun patch.
8. **Traçabilité des Datasets.**
   - `heron dataset describe` (appelé par `gen_dataset.sh`) enregistre dans `dataset.yaml` l'image
     qui a généré le Dataset : chemin, et label de version lu par `apptainer inspect`.
   - `heron prepare` refuse un Corpus qui mélange des Datasets d'images différentes, sauf si la
     recette contient `allow_mixed_images: true`.
   - Ajoute les tests pytest correspondants dans `tests/`. Toute la suite doit passer :
     `python -m pytest -q`.
9. **Test de la phase A**, dans le conteneur de développement. Lance la génération d'une scène
   synthétique (mode `synthetic`, m1013, leeloo, voxel 0,02, `DATASET_SIZE=1`), comme le fait
   `run_generation.sh`. Il faut obtenir un `.h5` qui contient `group/0/voxel_grid` et
   `group/0/reachability_map` en 152³, avec des valeurs comprises entre 0 et 1, et une carte non
   vide. La master map m1013 à 0,02 (`master_ik_data0.02.npz` et `.h5`) doit être fournie ; si tu
   ne l'as pas, demande-la.

## Phase B : construire l'image

1. **Fichiers** : `images/generation/Dockerfile` et `images/generation/build.sh`, dans CapaciNet.
   - Le Dockerfile commence par `ARG BASE_IMAGE` puis `FROM ${BASE_IMAGE}`. Il n'ajoute que ce qui
     est propre à Heron : HighFive (version épinglée, vérifie que `data_generation` compile avec),
     cnpy (commit épinglé), octomap, `ros-jazzy-pcl-ros`, les autres dépendances apt de
     `data_generation/CMakeLists.txt`, puis CapaciNet (au commit de la branche) et
     `colcon build --packages-select curobo_msgs data_generation` (avec les dépendances
     nécessaires).
   - Vérifie si `doosan-robot2` (branche Jazzy, s'il en existe une) est vraiment nécessaire. Dans
     cuRobo v2, la configuration m1013 vient de `curobo/content`. Ne l'ajoute que si la génération
     en a besoin.
   - `build.sh` est **non interactif**. Il clone `Lab-CORO/curobo_ros` au commit épinglé, construit
     `docker/x86_jazzy_prod.dockerfile` avec `TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0"` (A100, RTX 30xx,
     H100), construit l'image Heron avec `--build-arg BASE_IMAGE=…`, puis convertit l'image en
     `.sif` :
     `apptainer build --fakeroot <out>.sif docker-daemon://<image>:<tag>`. Les dossiers de cache et
     de travail d'Apptainer se règlent par variables (`APPTAINER_CACHEDIR`, `APPTAINER_TMPDIR`).
     Inspire-toi de `data_generation/docker/convert2apptainer.bash`, sans ses questions
     interactives.
   - Plateforme : amd64 seulement.
2. **Tout est épinglé** : l'image de base NGC, le commit `curobo_ros`, le commit `curobo_msgs` (via
   `my.repos` ou en direct), le tag cuRobo (`v0.8.0`, déjà fixé par curobo_ros), HighFive, cnpy, le
   commit CapaciNet et la version de `ros2-apt-source`. Aucun `git clone` sans `-b <tag>` ou
   `git checkout <commit>`.
3. **Contrat d'exécution** (à garder, et à écrire dans `docs/images.md`, section « Generation
   image », qui annonce justement ce contrat) :
   - chemins `/home/ros2_ws/install/data_generation/share/data_generation/{data,launch}` inchangés ;
   - mêmes variables `APPTAINERENV_*` ;
   - un `/tmp` où l'on peut écrire ;
   - le cache warp sous `$HOME/.cache/warp` ;
   - fonctionnement sous `apptainer exec -C --net --network=none --nv`, donc sans réseau à
     l'exécution : tout ce qui est téléchargé ou compilé au premier lancement (noyaux NVRTC de
     cuRobo, cache warp) doit soit être dans l'image, soit aller dans un dossier monté ;
   - `rmw_fastrtps_cpp` et `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`.
4. **Labels.** L'image porte des `LABEL` lisibles par `apptainer inspect` dans le `.sif` :
   `heron.image=generation`, `heron.ros=jazzy`, `heron.curobo=v0.8.0`,
   `heron.curobo_ros=<commit>`, `heron.curobo_msgs=<commit>`, `heron.capacinet=<commit>`,
   `heron.highfive=<version>`, `heron.cuda_arch=8.0;8.6;9.0` et `heron.built=<date>`.
   Fais afficher ces labels par `heron doctor` (ligne « generation image »), et sers-t'en pour la
   règle du mode `scene` (A.7).
5. **Test de fumée du `.sif`**, sur le poste :
   - `apptainer exec -C --nv <sif> python3 -c "import torch, curobo; print(torch.__version__, torch.cuda.is_available(), curobo.__version__)"` ;
   - puis la génération d'une scène synthétique sous
     `apptainer exec -C --net --network=none --nv`, avec les mêmes montages que `job.sbatch`.
     Le critère est le même qu'en A.9.

## Livrables

- La branche `feat/generation-jazzy` avec des commits locaux propres : portage, Open3D, variables,
  mode scene, traçabilité, image, doc. Tous les tests pytest passent.
- `images/generation/Dockerfile`, `images/generation/build.sh`, et `docs/images.md` à jour.
- Le `.sif` construit, avec sa taille et la sortie de `apptainer inspect`.
- Un compte rendu court :
  - les versions épinglées ;
  - la correspondance entre anciens et nouveaux services ;
  - ce qui a été testé, et comment ;
  - ce qui ne l'a pas été ;
  - tout écart avec ce prompt.

## Hors périmètre

L'image d'entraînement (l'entraînement tourne en venv sur le cluster), le Jetson et le portage du
mode `scene`.

## Après toi, sur le cluster (pour information)

Le `.sif` sera copié sur Narval avec `rsync`. On y générera alors la master map m1013 à 0,02
(mode `create_rm`) avec la nouvelle image, puis on la comparera à l'actuelle
(`group/0/reachability_map`, 200³). Critère d'acceptation : au moins 99 % des voxels dans la même
classe sur 51, sans quoi on cherche la cause avant de produire des Datasets. Ton compte rendu doit
permettre cette comparaison : dis comment lancer `create_rm` avec la nouvelle image.
