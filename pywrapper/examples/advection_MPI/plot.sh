#!/bin/bash

heat_plot() {
    SOURCE=$1
    OUTPUT=$2
    gnuplot <<-EOFMarker
      set terminal png
      set output "${OUTPUT}"
      set view map
      set cbrange [0:1]
      set palette defined (0 0 0 0.5, 1 0 0 1, 2 0 0.5 1, 3 0 1 1, 4 0.5 1 0.5, 5 1 1 0, 6 1 0.5 0, 7 1 0 0, 8 0.5 0 0)
      # set palette gray
      plot "${SOURCE}" matrix with image
EOFMarker
}

mkdir -p plots

echo "Note: The plots show only the part of the domain handled by the master node."

N=50
for i in $(seq 0 $(expr $N - 1))
do
    index=$(printf "%03d" $i)
    heat_plot "output/save_${index}.txt" "plots/plot_${index}.png"
done
