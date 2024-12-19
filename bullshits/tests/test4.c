#include <stdio.h>
#include <stdlib.h>
int main(void)
{
	int cm,kg,BMI;//BMI defined as float
	printf("請輸入身高(cm):%d",cm);
	scanf("%d",&cm);
    printf("請輸入體重(kg):%d",kg);
	scanf("%d",&kg);
	BMI=("cm/(kg*kg)"); //why you quote this? isn't it calculation?
	printf("BMI=:%d",BMI);
	

	

    system("pause");
    return 0;
}

