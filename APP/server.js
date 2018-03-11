// Required dependencies
var express = require("express"),
    bodyParser = require("body-parser"),
    morgan = require("morgan"),
    http = require("http"),
    winston = require('winston');

// Creation of http connection
var app = express();
var server = http.createServer(app);

/*
 * Morgan used to log requests to the console in developer's mode
 * Comment this line in production mode
 */
app.use(morgan('dev'));

// Enable json body parser
app.use(bodyParser.json({limit: '20mb'}));
app.use(bodyParser.urlencoded({limit: '20mb', extended: true}));

// Inject client and swagger configuration
app.use(express.static('./public'));

server.listen(8080, function () {
    winston.info("Server listening to PORT 8080");
});

module.exports = app;
