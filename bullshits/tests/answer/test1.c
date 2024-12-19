#include <stdio.h>
#include <stdlib.h>

int main(){
    char name[8];
    int id, age;
    char blood;
    int height;
    float weight = 0.0;

    //data input
    printf("請輸入名字：");
    scanf("%s",&name[6]);

    printf("請輸入學號：");
    scanf(" %d", &id);

    printf("請輸入身高：");
    scanf(" %d",&height);

    printf("請輸入體重：");
    scanf(" %f",&weight);

    printf("請輸入年齡：");
    scanf(" %d", &age);

    printf("請輸入血型：");
    scanf(" %c", &blood);

    //process data
    //      spaces
    printf("\n \n");
    printf("我的名字是      ：%s       我的學號是：%d ",name,id);
    printf("\n我的身高有      ：%d 公分  我的體重有：%f 公斤\n",height, weight);
    printf("我的年紀是      ：%d 歲    我的血型是：%c 型\n", (int)age, blood);

    system("pause");
    return 0;
}
