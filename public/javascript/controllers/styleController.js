angular.module('ucode18')

    .controller('styleCtrl', ['$scope', '$state', function ($scope) {

        $scope.styleImageView = "../images/style/view/original.jpg";

        $scope.showClothCard = false;
        $scope.showStyleCard = false;
        $scope.showLogoCard = false;

        $scope.clothesList = [
            {id:0,src:"../images/style/clothes/camisetabase.jpg", price: 20.5, selected: false, model: "XX1"},
            {id:1,src:"../images/style/clothes/camisetabasenegro.jpg", price: 1, selected: false, model: "XX3"}];
        $scope.stylesList = [
            {id:0,src:"../images/style/clothes/camisetalrayas.jpg", price: 1, selected: false, model: "XX4"}];
        $scope.logosList = [
            {id:0,src:"../images/style/clothes/camisetalogogrande.jpg", price: 5, selected: false, model: "XX4"},
            {id:1,src:"../images/style/clothes/camisetalogopequeno.jpg", price: 1, selected: false, model: "XX5"}];
        $scope.totalClothes = 0;
        $scope.totalStyles = 0;
        $scope.totalLogos = 0;

        $scope.activeCloth = function(id) {
            for (i = 0; i < $scope.clothesList.length; i++) {
                if (id === $scope.clothesList[i].id && $scope.clothesList[i].selected) {
                    $scope.clothesList[i].selected = false;
                    $scope.totalClothes = 0;
                } else if (id === $scope.clothesList[i].id && !$scope.clothesList[i].selected) {
                    $scope.clothesList[i].selected = true;
                    $scope.totalClothes = $scope.clothesList[i].price;
                } else {
                    $scope.clothesList[i].selected = false;
                }
            }
        };

        $scope.activeStyle = function(id) {
            for (i = 0; i < $scope.stylesList.length; i++) {
                if (id === $scope.stylesList[i].id && $scope.stylesList[i].selected) {
                    $scope.stylesList[i].selected = false;
                    $scope.totalStyles = 0;
                } else if (id === $scope.stylesList[i].id && !$scope.stylesList[i].selected) {
                    $scope.stylesList[i].selected = true;
                    $scope.totalStyles = $scope.stylesList[i].price;
                } else {
                    $scope.stylesList[i].selected = false;
                }
            }
        };

        $scope.activeLogo = function(id) {
            for (i = 0; i < $scope.logosList.length; i++) {
                if (id === $scope.logosList[i].id && $scope.logosList[i].selected) {
                    $scope.logosList[i].selected = false;
                    $scope.totalLogos = 0;
                } else if (id === $scope.logosList[i].id && !$scope.logosList[i].selected) {
                    $scope.logosList[i].selected = true;
                    $scope.totalLogos = $scope.logosList[i].price;
                } else {
                    $scope.logosList[i].selected = false;
                }
            }
        };

        $scope.openCard = function(card) {
            switch (card) {
                case 1:
                    $scope.showClothCard = !$scope.showClothCard;
                    $scope.showStyleCard = false;
                    $scope.showLogoCard = false;
                    break;
                case 2:
                    $scope.showClothCard = false;
                    $scope.showStyleCard = !$scope.showStyleCard;
                    $scope.showLogoCard = false;
                    break;
                case 3:
                    $scope.showClothCard = false;
                    $scope.showStyleCard = false;
                    $scope.showLogoCard = !$scope.showLogoCard;
                    break;
            }
        };

        $scope.changeImageView = function() {

        }

    }]);
