#!/bin/bash
#SBATCH -e /n/holyscratch01/zhuang_lab/Users/cosmosyw/DNA_MERFISH/Mecp2_1201_DNA_analysis/Logs/slurm-%j.err
#SBATCH -o /n/holyscratch01/zhuang_lab/Users/cosmosyw/DNA_MERFISH/Mecp2_1201_DNA_analysis/Logs/slurm-%j.out
#SBATCH -t 2-00:00:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p shared,zhuang

source activate spot_finder

sbatch -p zhuang -c 1 --mem 12000 -t 2-12 --wrap="SpotDNA -d /n/holyscratch01/zhuang_lab/Users/cosmosyw/DNA_MERFISH/20231221_Ma_DNA_from_1201/ -o /n/holyscratch01/zhuang_lab/Users/cosmosyw/DNA_MERFISH/Mecp2_1201_DNA_newSpotDNA/ --ref H0C1 -p /n/holylfs04/LABS/zhuang_lab/Users/cosmosyw/Spot_finder_files/picking_parameter_dict.pkl -c /n/holylfs04/LABS/zhuang_lab/Users/cosmosyw/Spot_finder_files/correction_dict_RC.pkl --fov 104 -m /n/holylfs04/LABS/zhuang_lab/Users/cosmosyw/Spot_finder_files/storm6_microscope.json -s /n/holylfs04/LABS/zhuang_lab/Users/cosmosyw/DNA_segment/Conv_zscan_104.npy" -o /n/holyscratch01/zhuang_lab/Users/cosmosyw/DNA_MERFISH/Mecp2_1201_DNA_newSpotDNA/Logs/SpotDNA_fov_104_output.txt
sleep 1
sbatch -p zhuang -c 1 --mem 12000 -t 2-12 --wrap="SpotDNA -d /n/holyscratch01/zhuang_lab/Users/cosmosyw/DNA_MERFISH/20231221_Ma_DNA_from_1201/ -o /n/holyscratch01/zhuang_lab/Users/cosmosyw/DNA_MERFISH/Mecp2_1201_DNA_newSpotDNA/ --ref H0C1 -p /n/holylfs04/LABS/zhuang_lab/Users/cosmosyw/Spot_finder_files/picking_parameter_dict.pkl -c /n/holylfs04/LABS/zhuang_lab/Users/cosmosyw/Spot_finder_files/correction_dict_RC.pkl --fov 196 -m /n/holylfs04/LABS/zhuang_lab/Users/cosmosyw/Spot_finder_files/storm6_microscope.json -s /n/holylfs04/LABS/zhuang_lab/Users/cosmosyw/DNA_segment/Conv_zscan_196.npy" -o /n/holyscratch01/zhuang_lab/Users/cosmosyw/DNA_MERFISH/Mecp2_1201_DNA_newSpotDNA/Logs/SpotDNA_fov_196_output.txt
sleep 1

sbatch -p zhuang -c 1 --mem 12000 -t 2-12 --wrap="SpotDNA -d /n/holyscratch01/zhuang_lab/Users/cosmosyw/DNA_MERFISH/20231221_Ma_DNA_from_1201/ -o /n/holyscratch01/zhuang_lab/Users/cosmosyw/DNA_MERFISH/Mecp2_1201_DNA_newSpotDNA/ --ref H0C1 -p /n/holylfs04/LABS/zhuang_lab/Users/cosmosyw/Spot_finder_files/picking_parameter_dict.pkl -c /n/holylfs04/LABS/zhuang_lab/Users/cosmosyw/Spot_finder_files/correction_dict_RC.pkl --fov 217 -m /n/holylfs04/LABS/zhuang_lab/Users/cosmosyw/Spot_finder_files/storm6_microscope.json -s /n/holylfs04/LABS/zhuang_lab/Users/cosmosyw/DNA_segment/Conv_zscan_217.npy" -o /n/holyscratch01/zhuang_lab/Users/cosmosyw/DNA_MERFISH/Mecp2_1201_DNA_newSpotDNA/Logs/SpotDNA_fov_217_output.txt
sleep 1

sbatch -p zhuang -c 1 --mem 12000 -t 2-12 --wrap="SpotDNA -d /n/holyscratch01/zhuang_lab/Users/cosmosyw/DNA_MERFISH/20231221_Ma_DNA_from_1201/ -o /n/holyscratch01/zhuang_lab/Users/cosmosyw/DNA_MERFISH/Mecp2_1201_DNA_newSpotDNA/ --ref H0C1 -p /n/holylfs04/LABS/zhuang_lab/Users/cosmosyw/Spot_finder_files/picking_parameter_dict.pkl -c /n/holylfs04/LABS/zhuang_lab/Users/cosmosyw/Spot_finder_files/correction_dict_RC.pkl --fov 256 -m /n/holylfs04/LABS/zhuang_lab/Users/cosmosyw/Spot_finder_files/storm6_microscope.json -s /n/holylfs04/LABS/zhuang_lab/Users/cosmosyw/DNA_segment/Conv_zscan_256.npy" -o /n/holyscratch01/zhuang_lab/Users/cosmosyw/DNA_MERFISH/Mecp2_1201_DNA_newSpotDNA/Logs/SpotDNA_fov_256_output.txt
sleep 1

sbatch -p zhuang -c 1 --mem 12000 -t 2-12 --wrap="SpotDNA -d /n/holyscratch01/zhuang_lab/Users/cosmosyw/DNA_MERFISH/20231221_Ma_DNA_from_1201/ -o /n/holyscratch01/zhuang_lab/Users/cosmosyw/DNA_MERFISH/Mecp2_1201_DNA_newSpotDNA/ --ref H0C1 -p /n/holylfs04/LABS/zhuang_lab/Users/cosmosyw/Spot_finder_files/picking_parameter_dict.pkl -c /n/holylfs04/LABS/zhuang_lab/Users/cosmosyw/Spot_finder_files/correction_dict_RC.pkl --fov 264 -m /n/holylfs04/LABS/zhuang_lab/Users/cosmosyw/Spot_finder_files/storm6_microscope.json -s /n/holylfs04/LABS/zhuang_lab/Users/cosmosyw/DNA_segment/Conv_zscan_264.npy" -o /n/holyscratch01/zhuang_lab/Users/cosmosyw/DNA_MERFISH/Mecp2_1201_DNA_newSpotDNA/Logs/SpotDNA_fov_264_output.txt
sleep 1
sbatch -p zhuang -c 1 --mem 12000 -t 2-12 --wrap="SpotDNA -d /n/holyscratch01/zhuang_lab/Users/cosmosyw/DNA_MERFISH/20231221_Ma_DNA_from_1201/ -o /n/holyscratch01/zhuang_lab/Users/cosmosyw/DNA_MERFISH/Mecp2_1201_DNA_newSpotDNA/ --ref H0C1 -p /n/holylfs04/LABS/zhuang_lab/Users/cosmosyw/Spot_finder_files/picking_parameter_dict.pkl -c /n/holylfs04/LABS/zhuang_lab/Users/cosmosyw/Spot_finder_files/correction_dict_RC.pkl --fov 265 -m /n/holylfs04/LABS/zhuang_lab/Users/cosmosyw/Spot_finder_files/storm6_microscope.json -s /n/holylfs04/LABS/zhuang_lab/Users/cosmosyw/DNA_segment/Conv_zscan_265.npy" -o /n/holyscratch01/zhuang_lab/Users/cosmosyw/DNA_MERFISH/Mecp2_1201_DNA_newSpotDNA/Logs/SpotDNA_fov_265_output.txt
sleep 1


echo Finish finding spots
