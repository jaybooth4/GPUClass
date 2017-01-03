#!/bin/bash

#############################################
#
# Execution script 
# ----------------
#
# by Julian Gutierrez
#
# Note: Run from main folder.
#############################################

#############################################
#
# Static configuration variables
#
#############################################

# Algorithm name
alg="sobel"

# Set tests name
image="synthetic.intensities.pgm"

#############################################
#
# On the run configuration variables
#
#############################################

# Message
echo "*******************************"
echo "* Running Final Project tests *"
echo "*******************************"
echo ""

# Get username
echo -n "Student username: "
read name 

# Get number of iterations
echo -n "How many iterations of the tests? Num: "
read iterations

#############################################
#
# Script execution
#
#############################################


echo "*********"
echo "* Running $image test"
echo "*********"
	
#Set input files: main/inputs/synthetic/synthetic.intensities.pgm
imagefolder="inputs/synthetic/$image"

#Set result folder: result/username/sobel
result="results/$name/$alg"
        
#Remove previous results: results/username/, make results/username/sobel
rm -rf results/$name/
mkdir -p results/$name/$alg

#compile cuda code: Makefile in: users/username/sobel/
echo "* Compiling Program *"
cd users/$name/$alg
make -s clean 
make -s all
cd ../../../

# Execute for specified iterations, executable: users/username/sobel/sobel                
for (( i=1; i<=$iterations; i++ ))
do			
         echo "* Running iteration $i"
         ./users/$name/$alg/$alg \
         --image $imagefolder > $result/$i.log
         wait
         mv result.ppm $result/result.ppm
done

# Go to re
cd $result/
	
# Grep all iterations and find Max, Min, Avg
grep "Kernel Execution Time" * | awk '{print $4}' > Kresultlist

#calculate max
Kmax=$(cat Kresultlist | sort -nr | head -n 1)
        
#calculate min
Kmin=$(cat Kresultlist | sort -n | head -n 1)
        
#calculate avg
Kavg=$(cat Kresultlist | awk '{ total += $1; count++} END {print total/count}')
        
grep "Total GPU Execution Time" * | awk '{print $5}' > Tresultlist
        
#calculate max
Tmax=$(cat Tresultlist | sort -nr | head -n 1)
        
#calculate min
Tmin=$(cat Tresultlist | sort -n | head -n 1)
        
#calculate avg
Tavg=$(cat Tresultlist | awk '{ total += $1; count++} END {print total/count}')

#Save to file
resultFile="$test.summary.csv"
	
echo "Test,Max Kernel,Min Kernel,Avg Kernel,Max GPU Time,Min GPU Time,Avg GPU Time" > $resultFile
echo "$test,$Kmax,$Kmin,$Kavg,$Tmax,$Tmin,$Tavg" >> $resultFile

echo "Results"
echo "-------"
echo "Average Kernel Time = $Kavg"
echo "Average Total GPU Execution Time = $Tavg"
       
#remove temp files
rm Kresultlist Tresultlist
        
#go back to main folder
cd ../../../
