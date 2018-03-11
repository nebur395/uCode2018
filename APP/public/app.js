angular.module('ucode18', ['ui.router', 'ngFileUpload'])

    .config(function ($stateProvider, $urlRouterProvider) {
        $stateProvider

        //starter screen
            .state('starter', {
                url: "/starter",
                templateUrl: "templates/starter.html"
            })

            //starter screen
            .state('style', {
                url: "/style",
                templateUrl: "templates/style.html",
                controller: "styleCtrl"
            });

        $urlRouterProvider.otherwise('starter');
    });
