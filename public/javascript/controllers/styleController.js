angular.module('ucode18')

    .controller('styleCtrl', ['$scope', '$state', function ($scope) {

        $scope.clothesList = [
            {id:0,src:"../images/style/clothes/camisetabase.jpg", price: 20.5, selected: false},
            {id:1,src:"../images/style/clothes/camisetabase.jpg", price: 36.5, selected: false}];
        $scope.stylesList = [
            {id:0,src:"../images/style/clothes/camisetabasenegro.jpg", price: 20.5, selected: false},
            {id:1,src:"../images/style/clothes/camisetalrayas.jpg", price: 36.5, selected: false}];
        $scope.logosList = [
            {id:0,src:"../images/style/clothes/camisetalogogrande.jpg", price: 20.5, selected: false},
            {id:1,src:"../images/style/clothes/camisetalogopequeno.jpg", price: 36.5, selected: false}];

        $scope.activeCloth = function(id) {
            for (i = 0; i < $scope.clothesList.length; i++) {
                $scope.clothesList[i].selected = id === $scope.clothesList[i].id;
            }
        }

    }]);
