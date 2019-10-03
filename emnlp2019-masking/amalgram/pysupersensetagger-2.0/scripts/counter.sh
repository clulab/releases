counter=0
for each in ./*.tags;
do
counter=`expr $counter+1`
done
echo "value of counter is:"$counter
