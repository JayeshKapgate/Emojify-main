var express = require('express');
var router = express.Router();
let {PythonShell} = require('python-shell')
var multer  = require('multer');
var path = require('path')

var storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, 'uploads/')
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + path.extname(file.originalname)) //Appending extension
  }
})
const upload = multer({storage:storage})

router.post('/plant', upload.single('image'), function(req, res, next) {

  try {
    
  
    var image = req.file.filename;

   
  // Configure the Python shell
  var options = {
    mode: 'text',
    pythonOptions: ['-u'], // unbuffered stdout
    scriptPath: 'routes/model/',
    args: ['-']
  };
  console.log("python")
  // Run the Python script and get the output
  var pyshell = new PythonShell('predict.py', options);
  pyshell.send(image);
  pyshell.on('message', function(message) {
    console.log(message);
    res.json({ disease: message });
  });
  pyshell.end(function (err) {
    if (err) {
      console.error(err);
      res.status(500).json({ error: 'An error occurred while running the prediction model.' });
    }
  });

} catch (error) {
    console.log(error)
}
});

module.exports = router;
