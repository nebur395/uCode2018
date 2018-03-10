angular.module('ucode18')

    .controller('styleCtrl', ['$scope', '$state', function ($scope) {

        $scope.clothesList = [
            {}];

        $scope.getNumber = function(num) {
            return new Array(num);
        }

    }]);
