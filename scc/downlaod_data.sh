#!/bin/bash -l
#$ -N bl_dl
#$ -P nsf-energize
#$ -l h_rt=12:00:00
#$ -j y
#$ -o download.log

cd /projectnb/nsf-energize/dgordon/Projects/BatteryLife/dataset

FILES=(
  "CALB.zip"
  "CALCE.zip"
  "HNEI.zip"
  "HUST.zip"
  "ISU_ILCC.zip"
  "Life%20labels.zip"
  "MATR.zip"
  "MICH.zip"
  "MICH_EXP.zip"
  "NA-ion.zip"
  "READMEs.zip"
  "RWTH.zip"
  "SDU.zip"
  "SNL.zip"
  "Stanford.zip"
  "Stanford_2.zip"
  "Tongji.zip"
  "UL_PUR.zip"
  "XJTU.zip"
  "ZN-coin.zip"
)

for f in "${FILES[@]}"; do
  name="${f//%20/ }"
  wget -c "https://zenodo.org/records/19688272/files/${f}?download=1" -O "${name}"
  if unzip -o "${name}" -d .; then
    rm -f "${name}"
  else
    echo "FAILED to unzip ${name} — keeping zip for retry" >&2
  fi
done