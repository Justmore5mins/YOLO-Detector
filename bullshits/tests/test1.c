#include <stdio.h>
#include <stdlib.h>
int main(void)
{
	char name[6],ABO;
	int number,high,weight,old; //weight用float定義相對合適
	
	
	printf("請輸入名字:");
	scanf("%c",&name); //%c是用在只有一個英文字元，要用%s
	
	printf("請輸入學號:");
	scanf("%d",&number);
	
	printf("請輸入身高:");
	scanf("%d",&high);
	
	printf("請輸入體重:");
	scanf("%d",&weight);
	
	printf("請輸入年齡:");
	scanf("%d",&old);
	
	printf("請輸入血型:");
	scanf("%c",&ABO);
	
	
	printf("我的名字是    :%c      我的學號是 :%d\n我的身高有    :%d公分 我的體重有 : %d 公斤\n我的年紀是 :%d 歲   我的血型是 :%c 型",name,number,high,weight,old,ABO);//格式化輸出，要有空格


    system("pause");
    return 0;
}