#include<GL/glut.h>
#include<stdlib.h>
#include<iostream>
using namespace std;
int x1,x2,y1,y2;
void display(void)
{
    int dx,dy,e;
    float x,y;
    dx=x2-x1;
    dy=y2-y1;
    e=2*dy-dx;
    x=x1;
    y=y1;

    glClear(GL_COLOR_BUFFER_BIT);

    glColor3f(0.0,1.0,0.0);
    glBegin(GL_POINTS);
    glVertex2i(x,y);

    int i;
    for(i=0;i<dx;i++)
    {
        if(e>0)
        {
            y=y+1;
            glVertex2i(x,y);
        }
        x=x+1;
        e=e+2*dy;
        glVertex2i(x,y);


    }
    glEnd();
    glFlush();
}
void init(void)
{
    glClearColor(0.0,0.0,0.0,0.0);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-100.0,100.0,-100.0,100.0,-1.0,1.0);
}
int main(int argc,char**argv)
{
    cout<<"Enter the value of x1,y1:";
    cin>>x1>>y1;
    cout<<"Enter the value of x2,y2:";
    cin>>x2>>y2;

    glutInit(&argc,argv);
    glutInitDisplayMode(GLUT_SINGLE|GLUT_RGB);
    glutInitWindowSize(500,500);
    glutInitWindowPosition(100,100);
    glutCreateWindow("Breshanman line algorithm");
    init();
    glutDisplayFunc(display);
    glutMainLoop();
    return 0;
}
