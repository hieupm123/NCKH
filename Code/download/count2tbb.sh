#!/bin/bash

PROGDIR=/datausers3/kttv/tien/ClassificationProject

WORKDIR="mtsat_data/working/$6/netcdf" 

# gcc count2tbb.c -o count2tbb
# gcc cnt_tbb.c -o cnt_tbb
cd "$WORKDIR" || { echo "Không thể vào thư mục $WORKDIR"; exit 1; }


for YYYY in $1 ; do
  for MM in $2 ; do
    for DD in $3 ; do
      for HH in $4 ; do
#      for HH in 00 ; do
       for mm in $5; do
          FOLDER="${YYYY}${MM}${DD}${HH}${mm}"
          mkdir -p "$FOLDER"
          # Đi vào thư mục
          cd "$FOLDER" || exit
      #  for mm in 00; do
#        mm=57
#        wget ftp://mtsat-1r.cr.chiba-u.ac.jp/pub/grid-MTSAT-2.0/MTSAT-1R/${YYYY}${MM}/${YYYY}${MM}${DD}${HH}??.ir.tar.bz2
        # wget ftp://mtsat.cr.chiba-u.ac.jp/MTSAT-2/gridded_V2.0/${YYYY}${MM}/${YYYY}${MM}${DD}${HH}${mm}.ir.tar.bz2

        # ftp://mtsat.cr.chiba-u.ac.jp/pub/grid-MTSAT-2.0/MTSAT-1R/
        FILE_NAME="${YYYY}${MM}${DD}${HH}${mm}.ir.tar.bz2"
        # wget ftp://mtsat-1r.cr.chiba-u.ac.jp/pub/grid-MTSAT-2.0/MTSAT-1R/${YYYY}${MM}/${YYYY}${MM}${DD}${HH}${mm}.ir.tar.bz2
        wget ftp://mtsat-1r.cr.chiba-u.ac.jp/pub/MTSAT-1R/gridded_V2.0/${YYYY}${MM}/${YYYY}${MM}${DD}${HH}${mm}.ir.tar.bz2

        if [ ! -e "$FILE_NAME" ]; then
            cd ..  # Quay lại thư mục cha
            rm -r "$FOLDER"  # Xóa thư mục nếu rỗng
            continue  # Bỏ qua vòng lặp này
          fi

        if [ ! \( -e ${YYYY}${MM}${DD}${HH}??.ir.tar.bz2 \) ] ; then
          echo "${YYYY}${MM}${DD}${HH}"
        else
          MI=`ls ${YYYY}${MM}${DD}${HH}??.ir.tar.bz2 | cut -c11-12`
#          mkdir -p ${YYYY}${MM}${DD}${HH}${MI}
          tar xvfj ${YYYY}${MM}${DD}${HH}${MI}.ir.tar.bz2 ; rm ${YYYY}${MM}${DD}${HH}${MI}.ir.tar.bz2
          for VAR in IR1 IR2 IR3 ; do
            var=`echo "${VAR}" | tr '[A-Z]' '[a-z]'`
            for NUM in 01 02 03 04 05 06 07 08 09 10 ; do
              if [ -f hdr_${var}_${YYYY}${MM}${DD}${HH}${MI}_0${NUM}.txt ] ; then
                awk '/^[0-9]:=/ || /^[0-9][0-9]:=/ || /^[0-9][0-9][0-9]:=/ || /^[0-9][0-9][0-9][0-9]:=/ || /^[0-9][0-9][0-9][0-9][0-9]:=/ {print}' hdr_ir1_${YYYY}${MM}${DD}${HH}${MI}_0${NUM}.txt | cut -d: -f1 > tmpa.txt
                awk '/^[0-9]:=/ || /^[0-9][0-9]:=/ || /^[0-9][0-9][0-9]:=/ || /^[0-9][0-9][0-9][0-9]:=/ || /^[0-9][0-9][0-9][0-9][0-9]:=/ {print}' hdr_ir1_${YYYY}${MM}${DD}${HH}${MI}_0${NUM}.txt | cut -d= -f2 > tmpb.txt
                paste tmpa.txt tmpb.txt > tbbtable.txt
              fi
            done
#            rm tmpa.txt
#            rm tmpb.txt

            # # Kiểm tra file DK01, DK02, DK03 tồn tại không
            for DK in "IMG_DK01${VAR}_${YYYY}${MM}${DD}${HH}${MI}.geoss"; do
                if [ -f "$DK" ]; then
                    echo "Tìm thấy file tồn tại: $DK"
                    dd if="$DK" of=tmp.geoss conv=swab
                    break  # Dừng vòng lặp khi tìm thấy file hợp lệ
                fi
            done
             $PROGDIR/count2tbb tmp.geoss
             mv tbb_tmp.geoss ${YYYY}${MM}${DD}${HH}${mm}_${VAR}bt.grd
             #convert to netcdf
             echo dset ${YYYY}${MM}${DD}${HH}${mm}_${VAR}bt.grd > mtsat.ctl
             echo title MTSATIR1 >> mtsat.ctl
             echo options  yrev little_endian >> mtsat.ctl
             echo undef -999.0 >> mtsat.ctl
             echo xdef 3000 linear 80.02  0.04 >> mtsat.ctl
             echo ydef 3000 linear -59.98 0.04 >> mtsat.ctl
             echo zdef 1 linear 1 1 >> mtsat.ctl
             echo tdef 1 linear 01JUN05 1hr >> mtsat.ctl
             echo vars 1 >> mtsat.ctl
             echo tbb 0 99 brightness temperature [K] >> mtsat.ctl
             echo endvars >> mtsat.ctl
             cdo -f nc import_binary mtsat.ctl ${YYYY}${MM}${DD}${HH}${mm}_${VAR}bt.nc
             cdo -sellonlatbox,$7,$8,$9,${10} ${YYYY}${MM}${DD}${HH}${mm}_${VAR}bt.nc conbaomini_${YYYY}${MM}${DD}${HH}${mm}_${VAR}bt.nc 
#             rm tmp.geoss
#             rm tbbtable.txt
#             rm hdr*.txt
#             rm *.geoss 
          done
        fi
        rm tmpa.txt
        rm tmpb.txt
        rm tmp.geoss
        rm tbbtable.txt
        rm hdr*.txt
        rm *.geoss 
        rm *.grd
        rm *.ctl
        cd ..
       done
      done
    done
  done
done