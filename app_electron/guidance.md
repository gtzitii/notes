# *Win桌面App开发（基于electron）*

## 一、前置条件

### 1.1 开发端

| 设备   | 操作系统  |
| ------ | --------- |
| 个人PC | Win10 x64 |

> ⚠️最好带有显卡

### 1.2 运行端

| 设备   | 操作系统  |
| ------ | --------- |
| 个人PC | Win10 x64 |

### 1.3 所需软件

| 软件      | 版本      | 链接                                          |
| --------- | --------- | --------------------------------------------- |
| `node.js` | `24.11.1` | [下载地址](https://nodejs.org/zh-cn/download) |
| `VS Code` | `1.106.3` | [下载地址](https://code.visualstudio.com/)    |

> ⚠️版本没有严格限制，一般默认最新版

### 1.4 所需技能

| 语言         | 程度 |
| ------------ | ---- |
| `HTML`       | 了解 |
| `CSS`        | 了解 |
| `JavaScript` | 熟悉 |

> ⚠️语言掌握程度无硬性要求，可以通过AI辅助编程，主要是要学会搭建环境

## 二、环境搭建

### 2.1 项目配置

#### 2.2.1 node.js安装

1. 通过[1.3小节](#1.3 所需软件)提供的链接下载`node.js`压缩包，并解压至`D:\dev\`目录下

   > 💡安装目录可自定义

   ![](./assets/Screenshot 2025-12-11 175801.png)

2. 添加`node.js`环境变量

   ![](./assets/Screenshot 2025-12-09 181910.png)

   ![](./assets/Screenshot 2025-12-09 181924.png)

   ![](./assets/Screenshot 2025-12-09 181934.png)

   ![](./assets/Screenshot 2025-12-11 194041.png)

   ![](./assets/Screenshot 2025-12-11 194300.png)

   ![](./assets/Screenshot 2025-12-11 194456.png)

   ![](./assets/Screenshot 2025-12-11 194326.png)

3. 在`CMD`中验证环境变量

   ![](./assets/Screenshot 2025-12-11 181046.png)

#### 2.2.2 项目配置

1. 创建工作目录`D:\dev\project\app_electron`，并从该目录进入`CMD`，在`CMD`中输入`npm init vite@latest`，用`vite`脚手架构建项目，项目名称为`demo_frontend`，框架为`Vue`，语言为`JavaScript`

   ![](./assets/Screenshot 2025-12-13 102404.png)

   > 💡项目名称可自定义，`CMD`中的警告信息不用理会，因为我加了一些其他配置

2. 通过[1.3小节](#1.3 所需软件)中的链接下载`VS Code`安装包，并安装至`D:\dev\`目录下

   > 💡安装目录可自定义

3. 在`VS Code`中打开`D:\dev\project\app_electron\demo`，项目结构如图

   ![](./assets/Screenshot 2025-12-13 102536.png)

4. 安装开发所需插件

   ![](./assets/Screenshot 2025-12-11 182309.png)

5. 打开终端，输入`npm install`安装模块，安装完成后会在项目目录中自动添加`node_modules`文件夹，里面包含前端运行的各种库函数

   ![](./assets/Screenshot 2025-12-13 102718.png)

6. 在终端输入`npm run dev`启动`vite`服务器，并通过浏览器加载前端页面

   ![](./assets/Screenshot 2025-12-13 102833.png)

   ![](./assets/Screenshot 2025-12-13 102844.png)

   > 💡服务器不是硬件机器，其实就是一段监听程序，在监听电脑中的某个端口，一旦有网络请求到达该端口，程序就负责处理请求

### 2.2 electron配置

1. 在`VS Code`中打开`demo`项目，在终端中按以下步骤安装`electron`相关工具，结果如下图

   ```shell
   #设置镜像加速源
   setx ELECTRON_MIRROR "https://npmmirror.com/mirrors/electron/"
   
   #重启终端！！！
   
   #安装electron
   npm install electron@38.1.2 --save-dev   
   #安装electron-forge
   npm install @electron-forge/cli --save-dev
   npm exec --package=@electron-forge/cli -c "electron-forge import"
   ```

   > ❗设置完镜像加速源要重启终端

   ![](./assets/Screenshot 2025-12-13 132303.png)

2. 修改端口为`3000`

   > 💡端口号可自定义

   ![](./assets/Screenshot 2025-12-13 123013.png)

3. 修改项目文件

   ![](./assets/Screenshot 2025-12-13 120333.png)

   ![](./assets/Screenshot 2025-12-13 121559.png)

   ![](./assets/Screenshot 2025-12-13 135923.png)

   > `main.js`的内容如下
   >
   > ```javascript
   > import { app, BrowserWindow } from 'electron';
   > 
   > const createWindow = () => {
   >     const mainWindow = new BrowserWindow({
   >           width: 800,
   >            height: 600,
   >        });
   >       mainWindow.loadURL('http://localhost:3000');
   > };
   >   
   >    app.whenReady().then(() => {
   >       createWindow();
   >    });
   >   
   > app.on('window-all-closed', () => {
   >     if (process.platform !== 'darwin') {
   >         app.quit();
   >       }
   > });
   > ```

   ![](./assets/Screenshot 2025-12-13 140033.png)

   ![](./assets/Screenshot 2025-12-13 135349.png)

   > ```javascript
   > import { FusesPlugin } from '@electron-forge/plugin-fuses';
   > import { FuseV1Options, FuseVersion } from '@electron/fuses';
   > 
   > export default{
   > ```
   >
   > 💡所有修改的文件记得保存

4. 启动`electron`

   ![](./assets/Screenshot 2025-12-13 140150.png)

   ![](./assets/Screenshot 2025-12-13 140212.png)

   ![](./assets/Screenshot 2025-12-13 140236.png)

   > 💡`electron`是一种基于`web`技术的桌面`app`开发框架

## 三、功能开发

## 四、生产环境打包

### 4.1 生成可执行文件

1. 修改文件

   ![](./assets/Screenshot 2025-12-13 143900.png)

   ```javascript
   import path from 'path';
   import { fileURLToPath } from 'url';
   const __dirname = path.dirname(fileURLToPath(import.meta.url));
   ```

   ```javascript
   mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
   ```

   ![](./assets/Screenshot 2025-12-13 142450.png)

2. 构建可执行程序

   ![](./assets/Screenshot 2025-12-13 143047.png)

   ![](./assets/Screenshot 2025-12-13 143431.png)

   ![](./assets/Screenshot 2025-12-13 144100.png)

### 4.2 构建安装包

修改文件并通过`npm run make`构建安装包

![](./assets/Screenshot 2025-12-13 144457.png)

```json
  "author": "zzt",
  "description": "My Electron demo app",
```

![](./assets/Screenshot 2025-12-13 144640.png)

![](./assets/Screenshot 2025-12-13 144721.png)

![](./assets/Screenshot 2025-12-13 144734.png)

> 💡安装包默认安装到`C:\Users\用户\AppData\Local\demo\app-0.0.0`

## 五、拓展

- app包含其他资源

  ![](./assets/Screenshot 2025-12-13 150218.png)