

// HACK should be chrome.action since Manifest v3, but sometime(?) its still browserAction instead
(chrome.action || chrome.browserAction).onClicked.addListener(function(extensionTab) {

	var manifestData = chrome.runtime.getManifest();
	console.debug('starting tab for ' + manifestData.name + ' v' + manifestData.version + ' (Tab ID '+extensionTab.id+')...');

	//create tab and load main index.html
	chrome.tabs.create({
			url: '/index.html'
		}, function (viewerTab){
			var manifestData = chrome.runtime.getManifest();
			console.info('did create tab for ' + manifestData.name + ' v' + manifestData.version + ' (Tab ID '+viewerTab.id+')');

		}
	);

});
