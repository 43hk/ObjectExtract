#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QFileDialog>
#include <QScreen>
#include <QMessageBox>
#include "imagepool.h"
#include "cvfunction.h"

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
    void resizeEvent(QResizeEvent *event);
    void imageDisplay();
    void refDisplay();

    bool isRGB = true;

private slots:
    void do_loadImage();
    void do_saveImage();
    void do_loadRef();
    void do_loadRefFromCam();
    void do_loadImageFromCam();
    void do_templateSearch();
    void do_startTracing();
    void do_faceSearch();
    void do_edgeDetection();
    void do_thresholding();

private:
    Ui::MainWindow *ui;
    std::unique_ptr<ImagePool> imageData;

    QString originalImagePath;
};
#endif // MAINWINDOW_H
