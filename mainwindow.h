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
    void imageRefresh();
    void imageDisplay();

private slots:
    void do_loadImage();
    void do_saveImage();
    void do_capture();
    void do_loadRef();
    void do_search();

private:
    Ui::MainWindow *ui;
    std::unique_ptr<ImagePool> imageData;

    QString originalImagePath;
    QImage myImage;
};
#endif // MAINWINDOW_H
