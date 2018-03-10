angular.module('ucode18')

    .controller('starterCtrl', ['$scope', '$state', function ($scope, $state) {

        $scope.firstView = true;
        $scope.secondView = false;
        $scope.thirdView = false;

        $scope.processing = function () {
            $scope.firstView = false;
            $scope.secondView = true;
            $scope.thirdView = false;
        };

        $scope.processed = function () {
            $scope.firstView = false;
            $scope.secondView = false;
            $scope.thirdView = true;
        };

        $scope.restartViews = function () {
            $scope.firstView = true;
            $scope.secondView = false;
            $scope.thirdView = false;
        };

        $scope.startProcess = function () {
            $scope.processing();
        };

    }]);
