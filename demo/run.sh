num=`wc -l words2do.txt |cut -f 1 -d ' '`
seq 1 $num | parallel -j 20 --eta sh best_window_size.sh {} &
