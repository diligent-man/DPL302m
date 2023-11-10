#!bin/bash

counter=1
for file in file_{1..1620}.txt
  do
    new_name=$counter.txt
    mv -- "$file" "$new_name"
    counter=$((counter+1))
done
