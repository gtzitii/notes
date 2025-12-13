import { app, BrowserWindow } from 'electron';

import path from 'path';
import { fileURLToPath } from 'url';
const __dirname = path.dirname(fileURLToPath(import.meta.url));

const createWindow = () => {
    const mainWindow = new BrowserWindow({
        width: 800,
        height: 600,
    });
    // mainWindow.loadURL('http://localhost:3000');
    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
};

app.whenReady().then(() => {
    createWindow();
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        app.quit();
    }
});