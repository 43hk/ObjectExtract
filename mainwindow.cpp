#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , imageData(std::make_unique<ImagePool>())
{
    ui->setupUi(this);
    ui->EditGroup->setVisible(false);

    connect(ui->actionLoad, &QAction::triggered, this, &MainWindow::do_loadImage);
    connect(ui->actionSave, &QAction::triggered, this, &MainWindow::do_saveImage);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::resizeEvent(QResizeEvent *event)
{
    QMainWindow::resizeEvent(event);
    imageRefresh(); // 窗口大小变化时更新图片显示
}

void MainWindow::imageRefresh()
{
    myImage = QImage((const unsigned char*)(imageData->dst.data),
                     imageData->dst.cols, imageData->dst.rows,
                     imageData->dst.step,
                     QImage::Format_RGB888).copy();

    // 获取 QLabel 当前尺寸
    QSize labelSize = ui->image->size();
    // 原始图片的宽高比=
    qreal imageAspectRatio = myImage.width() / (qreal)myImage.height();
    // 计算缩放后的尺寸，保持宽高比
    QSize scaledSize;
    if (imageAspectRatio > labelSize.width() / (qreal)labelSize.height())
    {
        // 如果图片更“扁”，以宽度为基准缩放
        scaledSize.setWidth(labelSize.width());
        scaledSize.setHeight(labelSize.width() / imageAspectRatio);
    }
    else
    {
        // 图片更“窄”，以高度为基准缩放
        scaledSize.setHeight(labelSize.height());
        scaledSize.setWidth(labelSize.height() * imageAspectRatio);
    }

    // 确保缩放后的尺寸不超过 QLabel 的大小
    scaledSize = scaledSize.boundedTo(labelSize);

    QPixmap scaledPixmap = QPixmap::fromImage(myImage).scaled(scaledSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);

    ui->image->setPixmap(scaledPixmap);
}

void MainWindow::imageDisplay()
{
    QImage myImage(
        (const unsigned char*)(imageData->dst.data), // 数据指针
        imageData->dst.cols,                         // 宽度
        imageData->dst.rows,                         // 高度
        imageData->dst.step,                         // 每行字节数
        QImage::Format_RGB888                  // 像素格式
        );
    QPixmap pixmap = QPixmap::fromImage(myImage);

    // 缩放图片以适应 QLabel 的大小
    QSize labelSize = ui->image->size(); // 获取 QLabel 的大小
    QPixmap scaledPixmap = pixmap.scaled(labelSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);

    ui->image->setPixmap(scaledPixmap);
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
            imageData->dst = imageData->src;
            imageDisplay();
            ui->EditGroup->setVisible(true);
            qDebug() << "Selected file path:" << originalImagePath;
        }
        else {qDebug() << "Failed to load image.";}
    }
}

void MainWindow::do_saveImage()
{
    QString baseName = QFileInfo(originalImagePath).completeBaseName(); // 获取不带扩展名的文件名

    // 设置过滤器以显示不同格式的图片类型
    QString filter = "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)";
    QString selectedFilter = "PNG (*.png)"; // 默认选中的过滤器

    // 显示保存对话框
    QString filename = QFileDialog::getSaveFileName(this, tr("保存图片"), baseName, filter, &selectedFilter);

    if (!filename.isEmpty())
    {
        // 根据用户选择的过滤器确定文件格式并更新文件名后缀
        if (selectedFilter.contains("PNG"))
            filename += ".png";
        else if (selectedFilter.contains("JPEG") || selectedFilter.contains("JPG"))
            filename += ".jpg";
        else if (selectedFilter.contains("BMP"))
            filename += ".bmp";

        bool saved = myImage.save(filename);

        if (saved)
            QMessageBox::information(this, tr("保存成功"), tr("图片已成功保存为: ") + filename);
        else
            QMessageBox::warning(this, tr("保存失败"), tr("无法保存图片，请检查文件路径和权限"));
    }
}
