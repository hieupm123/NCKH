#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define N150 150

int main(int argc, char *argv[]){
  int   lon_tl=20,  lat_tl=60;
  int   lon_br=140, lat_br=-60;
  float lon_inc=0.04;
  float lat_inc=0.04;

  FILE	*fp;
  char	outputfilename[N150];
  float tmp_NX, tmp_NY;
  int   i, j, k, nk, NX, NY, tmp_count, count[N150];
  short	*data_all;
  float	*dt, tmp_tbb, tbb[N150];

  if(argc!=2){
     fprintf(stderr,"Usage: geoss2bt.c inputfilename1\n");
     fprintf(stderr," e.g.: ./geoss2bt IMG_DK01IR1_200705010030.geoss\n");
     exit(1);
  }
  tmp_NX=(lon_br-lon_tl)/lon_inc;  tmp_NY=(lat_tl-lat_br)/lat_inc;
  NX=(int)(tmp_NX);  NY=(int)(tmp_NY);
  data_all=(short*)malloc(2*NY*NX);
  dt=(float*)malloc(4*NY*NX);
  for(i=0;i<NY*NX;i++){
     data_all[i]=-999;
     dt[i]=-999.;
  }

  if((fp=fopen(argv[1],"r"))==NULL){
     fprintf(stderr,"*** input file (%s) cannot open ***\n",argv[1]);
     exit(1);
  }
  fread(data_all,sizeof(short),NY*NX,fp);
  fclose(fp);

  if((fp=fopen("tbbtable.txt","r"))==NULL){
     fprintf(stderr,"*** TBB table (tbbtable.txt) cannot open ***\n");
     exit(1);
  }
  k=0;
  while(!feof(fp)){
     fscanf(fp,"%d %f",&tmp_count, &tmp_tbb);
     count[k]=tmp_count;  tbb[k]=tmp_tbb;
     k++;
  }
  nk=k;
  fclose(fp);

  for(i=0;i<NY;i++){
  for(j=0;j<NX;j++){
     if(data_all[NX*i+j]<count[0]){ dt[NX*i+j]=-999.; }
     else if(data_all[NX*i+j]==count[0]) { dt[NX*i+j]=tbb[0]; }
     else{
     	for(k=1;k<nk;k++){
     	   if(data_all[NX*i+j]<=count[k]){
     	   	dt[NX*i+j]=tbb[k]+(tbb[k]-tbb[k-1])*(count[k]-data_all[NX*i+j])/(count[k]-count[k-1]);
     	   	goto LOOP;
     	   }
     	}
     }
     LOOP:;
  }}

  sprintf(outputfilename,"tbb_%s",argv[1]);
  fopen(outputfilename,"w");
  fwrite(dt,sizeof(float),NX*NY,fp);
  fclose(fp);

  free(data_all);  free(dt);
  return 0;
}
