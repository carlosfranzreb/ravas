
const path = require('path');
const fs = require('fs-extra');

const extTargetDir = path.resolve(__dirname, '..', '..', 'dist', 'chrome-extension');
const extResDir = path.resolve(__dirname, '..', 'chrome-extension');
const extSrcDir = path.resolve(__dirname, '..', '..', 'build');

fs.emptyDir(extTargetDir).then(() => Promise.all([
  fs.copy(extSrcDir, extTargetDir),
  fs.copy(extResDir, extTargetDir)
])).then(() => console.log('copied chrome-extension to ', extTargetDir));
