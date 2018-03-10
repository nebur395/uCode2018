angular.module('ucode18')

    //include the 'cloth.html' into the <cloth> tag
    .directive('search', function () {
        return {
            restrict: 'E',
            templateUrl: 'templates/components/search.html',
            controller: 'searchCtrl',
            scope: {}
        }
    });
