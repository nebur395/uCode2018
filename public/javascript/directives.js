angular.module('ucode18')

// include the 'navbar.html' into the <navbar> tag
    .directive('description', function () {
        return {
            restrict: 'E',
            templateUrl: 'templates/components/description.html',
            controller: 'descriptionCtrl',
            scope: {}
        }
    })

    //include the 'cloth.html' into the <cloth> tag
    .directive('search', function () {
        return {
            restrict: 'E',
            templateUrl: 'templates/components/search.html',
            controller: 'searchCtrl',
            scope: {}
        }
    });
