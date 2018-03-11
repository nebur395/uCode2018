angular.module('ucode18')

    .controller('styleCtrl', ['$scope', 'Upload', '$timeout', '$state', function ($scope, Upload, $timeout, $state) {

        $scope.progressPercentage = 0;

        $scope.$watch('files', function () {
            $scope.upload($scope.files);
        });
        $scope.$watch('file', function () {
            if ($scope.file != null) {
                $scope.files = [$scope.file];
            }
        });
        $scope.upload = function (files) {
            if (files && files.length) {
                for (var i = 0; i < files.length; i++) {
                    var file = files[i];
                    if (!file.$error) {
                        Upload.upload({
                            url: 'https://angular-file-upload-cors-srv.appspot.com/upload',
                            data: {
                                username: $scope.username,
                                file: file
                            }
                        }).then(function (resp) {
                            $timeout(function() {
                            });
                        }, null, function (evt) {
                            $scope.progressPercentage = parseInt(100.0 *
                                evt.loaded / evt.total);
                            console.log($scope.progressPercentage);
                        });
                    }
                }
            }
        };

        $scope.showOptions = false;

        $scope.$watch('progressPercentage', function () {
            if ($scope.progressPercentage === 100) {
                $scope.showOptions = true;
            }
        });

        $scope.styleImageView = "../images/style/view/original.jpg";
        $scope.dropBackgroundImage = "../images/style/drop.png";

        $scope.showClothCard = false;
        $scope.showStyleCard = false;
        $scope.showLogoCard = false;

        $scope.clothesList = [
            {id:0,src:"../images/style/clothes/camisetabase.jpg", price: 20, selected: false, model: "XX1"},
            {id:1,src:"../images/style/clothes/camisetabasenegro.jpg", price: 25, selected: false, model: "XX3"}];
        $scope.stylesList = [
            {id:0,src:"../images/style/clothes/camisetalrayas.jpg", price: 5, selected: false, model: "XX4"}];
        $scope.logosList = [
            {id:0,src:"../images/style/clothes/camisetalogogrande.jpg", price: 10, selected: false, model: "XX4"},
            {id:1,src:"../images/style/clothes/camisetalogopequeno.jpg", price: 5, selected: true, model: "XX5"}];
        $scope.totalToPay = $scope.logosList[1].price;
        $scope.totalClothes = 0;
        $scope.totalStyles = 0;
        $scope.totalLogos = $scope.logosList[1].price;

        $scope.activeCloth = function(id) {
            for (i = 0; i < $scope.clothesList.length; i++) {
                if (id === $scope.clothesList[i].id && $scope.clothesList[i].selected) {
                    $scope.clothesList[i].selected = false;
                    $scope.totalClothes = 0;
                    $scope.totalToPay = $scope.totalClothes + $scope.totalStyles + $scope.totalLogos;
                } else if (id === $scope.clothesList[i].id && !$scope.clothesList[i].selected) {
                    $scope.clothesList[i].selected = true;
                    $scope.totalClothes = $scope.clothesList[i].price;
                    $scope.totalToPay = $scope.totalClothes + $scope.totalStyles + $scope.totalLogos;
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
                    $scope.totalToPay = $scope.totalClothes + $scope.totalStyles + $scope.totalLogos;
                } else if (id === $scope.stylesList[i].id && !$scope.stylesList[i].selected) {
                    $scope.stylesList[i].selected = true;
                    $scope.totalStyles = $scope.stylesList[i].price;
                    $scope.totalToPay = $scope.totalClothes + $scope.totalStyles + $scope.totalLogos;
                } else {
                    $scope.stylesList[i].selected = false;
                }
            }
        };

        $scope.activeLogo = function(id) {
            for (i = 0; i < $scope.logosList.length; i++) {
                if (id === $scope.logosList[i].id && $scope.logosList[i].selected) {
                    $scope.totalLogos = 0;
                    $scope.totalToPay = $scope.totalClothes + $scope.totalStyles + $scope.totalLogos;
                } else if (id === $scope.logosList[i].id && !$scope.logosList[i].selected) {
                    $scope.logosList[i].selected = true;
                    $scope.totalLogos = $scope.logosList[i].price;
                    $scope.totalToPay = $scope.totalClothes + $scope.totalStyles + $scope.totalLogos;
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

        };

        $scope.goStarter = function () {
            $state.go('starter');
        }

    }]);
