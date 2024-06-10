a=int(input("enter the number of terms"))
n1,n2=0,1
count=0
if a <= 0:
    print("please enter a positive number")
elif a==1:
    print("fibonacci sequence upto",a,"term:")
    print(n1)
else:
    print("fibonacci sequence:")
    while count<a:
        print(n1)
        nth=n1+n2
        n1=n2
        n2=nth
        count +=1
