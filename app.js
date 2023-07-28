const express = require('express')
const app = express()
const port = 3000
var mongoose = require("mongoose")
const MongoClient = require('mongodb').MongoClient;
mongoose.connect('mongodb+srv://Elbin:ELBIN123@cluster0.r8yxmlm.mongodb.net/TEST?retryWrites=true&w=majority',{
    useNewUrlParser: true,
    useUnifiedTopology: true
});





const bodyParser = require('body-parser');
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
app.use(express.static('public'));
var db = mongoose.connection;

db.on('error',()=>console.log("Error in Connecting to Database"));
db.once('open',()=>console.log("Connected to Database"))

const collection=db.collection('users')
app.set('view engine', 'ejs')

app.get('/', (req, res) => {
    res.render('pages/index')
})
app.get('/login', (req, res) => {
    res.render('pages/login')
})

app.get('/text', (req, res) => {
  res.render('pages/text')
})

app.get('/signup', (req, res) => {
    res.render('pages/signup')
})

app.listen(port, () => {
  console.log(`App listening at port ${port}`)
})


app.post('/signup', function(req, res) {
    var name = req.body.name;
    var email = req.body.email;
    var password = req.body.password;
    var repeatPassword = req.body.repeatPassword;
    var response = false;
    // Insert data into MongoDB collection
    if(password != repeatPassword){
        res.render('pages/signup', {error: 'Passwords do not match'});
    }else{
    const data = {
        name: name,
        email: email,
        password: password,
    }
    db.collection('users').insertOne(data, function(err, result) {
        if (err) throw err;
        console.log(data);
        console.log("user added"); 
        console.log(result)
        res.send({success: true});
    });
    }
  });
  

app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).send('Something broke!');
  });

  app.post('/login1', async (req, res) => {
    console.log(req.body);
    try {
      let user;
      try {
        user = await collection.findOne({ email: req.body.email });
      } catch (error) {
        console.error('Error during findOne', error);
      }
      if (!user) {
        return res.status(400).send('User not found');
      }else{
        console.log("User found")
      }
      if (user.password !== req.body.password) {
        return res.status(400).send('Invalid password');
      }
      res.status(200).send({success: true, name: user.name});
      console.log(user.name);
    } catch (error) {
      res.status(500).send(error);
    }
  });