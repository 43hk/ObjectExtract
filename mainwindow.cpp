#include "mainwindow.h"
#include "ui_mainwindow.h"
using namespace cv;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , imageData(std::make_unique<ImagePool>())
{
    ui->setupUi(this);
    ui->EditGroup->setVisible(false);

    // File
    connect(ui->actionLoad,         &QAction::triggered, this, &MainWindow::do_loadImage);
    connect(ui->actionSave,         &QAction::triggered, this, &MainWindow::do_saveImage);
    connect(ui->actionReference,    &QAction::triggered, this, &MainWindow::do_loadRef);
    // Caputure
    connect(ui->actionReference_2,  &QAction::triggered, this, &MainWindow::do_loadRefFromCam);
    connect(ui->actionMain,         &QAction::triggered, this, &MainWindow::do_loadImageFromCam);


    connect(ui->SearchButton,       &QPushButton::clicked, this, &MainWindow::do_templateSearch);
    connect(ui->StartTrackingButton,&QPushButton::clicked, this, &MainWindow::do_startTracing);
    connect(ui->SearchFaceButton,   &QPushButton::clicked, this, &MainWindow::do_faceSearch);

    connect(ui->EdgeDetectionButton,&QPushButton::clicked, this, &MainWindow::do_edgeDetection);
    connect(ui->ThresholdingButton, &QPushButton::clicked, this, &MainWindow::do_thresholding);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::resizeEvent(QResizeEvent *event)
{
    QMainWindow::resizeEvent(event);
    imageDisplay(); // 窗口大小变化时更新图片显示
    refDisplay();
}

void MainWindow::imageDisplay()
{
    QImage myImage(
        (const unsigned char*)(imageData->dst.data),
        imageData->dst.cols,
        imageData->dst.rows,
        imageData->dst.step,
        QImage::Format_RGB888
        );
    QPixmap pixmap = QPixmap::fromImage(myImage);

    // 缩放图片以适应 QLabel 的大小
    QSize labelSize = ui->image->size(); // 获取 QLabel 的大小
    QPixmap scaledPixmap = pixmap.scaled(labelSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);

    ui->image->setPixmap(scaledPixmap);
}

void MainWindow::refDisplay()
{
    QImage myImage(
        (const unsigned char*)(imageData->ref.data),
        imageData->ref.cols,
        imageData->ref.rows,
        imageData->ref.step,
        QImage::Format_RGB888
        );
    QPixmap pixmap = QPixmap::fromImage(myImage);

    // 缩放图片以适应 QLabel 的大小
    QSize labelSize = ui->imageRef->size(); // 获取 QLabel 的大小
    QPixmap scaledPixmap = pixmap.scaled(labelSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    ui->imageRef->setPixmap(scaledPixmap);
}

void MainWindow::do_loadImage()
{
    originalImagePath = QFileDialog::getOpenFileName(this, tr("打开图片"), "", tr("图片文件 (*.png *.jpg *.jpeg *.bmp);;All Files (*)"));
    if (!originalImagePath.isEmpty())
    {
        QPixmap pixmap(originalImagePath);
        if (!pixmap.isNull())
        {
            ui->image->clear();

            imageData->src = imread(originalImagePath.toStdString());
            if(imageData->src.empty())
            {
                qDebug() << "Error: Failed to load image data.";
                return;
            }

            cvtColor(imageData->src, imageData->src, COLOR_BGR2RGB);
            imageData->dst = imageData->src.clone();
            ui->EditGroup->setVisible(true);
            imageDisplay();
            qDebug() << "Selected file path:" << originalImagePath;
        }
        else {qDebug() << "Failed to load image.";}
    }
}

void MainWindow::do_saveImage()
{
    QImage cutExport(
        (const unsigned char*)(imageData->cut.data),
        imageData->cut.cols,
        imageData->cut.rows,
        imageData->cut.step,
        QImage::Format_RGB888
        );

    QString baseName = QFileInfo(originalImagePath).completeBaseName(); // 获取不带扩展名的文件名

    // 设置过滤器以显示不同格式的图片类型
    QString filter = "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)";
    QString selectedFilter = "PNG (*.png)"; // 默认选中的过滤器

    // 显示保存对话框
    QString filename = QFileDialog::getSaveFileName(this, tr("保存图片"), baseName, filter, &selectedFilter);

    if (!filename.isEmpty())
    {
        bool saved =cutExport.save(filename);
        if(!saved)
            QMessageBox::warning(this, tr("保存失败"), tr("无法保存图片，请检查文件路径和权限"));
    }
}

void MainWindow::do_loadRef()
{
    imageData->dst = imageData->src.clone();

    originalImagePath = QFileDialog::getOpenFileName(this, tr("打开图片"), "", tr("图片文件 (*.png *.jpg *.jpeg *.bmp);;All Files (*)"));
    if (!originalImagePath.isEmpty())
    {
        QPixmap pixmap(originalImagePath);
        if (!pixmap.isNull())
        {
            imageData->ref = imread(originalImagePath.toStdString());
            if(imageData->ref.empty())
            {
                qDebug() << "Error: Failed to load reference.";
                return;
            }
            ui->EditGroup->setVisible(true);
            cvtColor(imageData->ref, imageData->ref, COLOR_BGR2RGB);
            refDisplay();
            qDebug() << "Selected ref file path:" << originalImagePath;
        }
        else {qDebug() << "Failed to load ref.";}
    }


}

void MainWindow::do_loadRefFromCam()
{
    imageData->dst = imageData->src.clone();

    VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open camera." << std::endl;
        return;
    }

    Mat frame;  // 用于存储捕获的图像

    while (true)
    {
        cap >> frame;  // 从摄像头读取一帧
        if (frame.empty())
        {
            std::cerr << "Error: Failed to capture image." << std::endl;
            break;
        }
        cv::imshow("Camera Feed", frame);
        // 等待用户按键捕获图像
        if (cv::waitKey(30) >= 0) break;
    }


    cap.release();
    cv::destroyAllWindows();

    imageData->ref = frame.clone();
    imshow("Reference", imageData->ref);
    cvtColor(imageData->ref, imageData->ref, COLOR_BGR2RGB);

    refDisplay();
    ui->EditGroup->setVisible(true);
}

void MainWindow::do_loadImageFromCam()
{
    ui->image->clear();
    VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open camera." << std::endl;
        return;
    }

    Mat frame;  // 用于存储捕获的图像

    while (true)
    {
        cap >> frame;  // 从摄像头读取一帧
        if (frame.empty())
        {
            std::cerr << "Error: Failed to capture image." << std::endl;
            break;
        }

        cv::imshow("Camera Feed", frame);

        // 等待用户按键捕获图像
        if (cv::waitKey(30) >= 0) break;
    }


    cap.release();
    destroyAllWindows();

    imageData->src = frame.clone();
    cvtColor(imageData->src, imageData->src, COLOR_BGR2RGB);
    imageData->dst = imageData->src.clone();

    ui->EditGroup->setVisible(true);
    imageDisplay();
}

void MainWindow::do_templateSearch()
{
    if(imageData->src.empty()) return;
    imageData->dst = imageData->src.clone();
    imageData->cut = imageData->src.clone();
    ui->image->clear();

    Method METHOD;
    if      (ui->CCORRButton->isChecked())  METHOD = Method::TM_CCORR;
    else if (ui->CCOEFFButton->isChecked()) METHOD = Method::TM_CCOEFF;
    else    METHOD = Method::TM_SQDIFF;

    if (imageData->ref.empty())
    {
        std::cerr << "Error: ref is empty!" << std::endl;
        return;
    }
    if (imageData->ref.cols > imageData->src.cols || imageData->ref.rows > imageData->src.rows)
    {
        std::cerr << "Error: Template image is larger than source image!" << std::endl;
        return;
    }

    ui->image->clear();

    Mat cutRes;
    cutRes = CVFunction::templateSearch(imageData->src, imageData->ref, imageData->dst, METHOD);
    imageDisplay();
    imageData->cut = cutRes.clone();
}

void MainWindow::do_startTracing()
{
    CVFunction::track(imageData->ref);
}

void MainWindow::do_faceSearch()
{
    if(imageData->src.empty()) return;
    imageData->dst = imageData->src.clone();
    imageData->cut = imageData->src.clone();
    ui->image->clear();

    Mat cutRes;
    cutRes = CVFunction::faceSearch(imageData->src, imageData->dst);
    imageDisplay();
    imageData->cut = cutRes.clone();
}

void MainWindow::do_edgeDetection()
{
    if(imageData->src.empty()) return;
    imageData->dst = imageData->src.clone();
    imageData->cut = imageData->src.clone();
    ui->image->clear();

    Mat cutRes;
    cutRes = CVFunction::edgeDetection(imageData->src, imageData->dst, 3);
    imageDisplay();
    imageData->cut = cutRes.clone();
}

void MainWindow::do_thresholding()
{
    if(imageData->src.empty()) return;
    imageData->dst = imageData->src.clone();
    imageData->cut = imageData->src.clone();
    ui->image->clear();

    Mat cutRes;
    cutRes = CVFunction::grabcutForegroundExtraction(imageData->src, imageData->dst);
    imageDisplay();
    imageData->cut = cutRes.clone();
}




