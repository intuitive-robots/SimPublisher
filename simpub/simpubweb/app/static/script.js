document.addEventListener('DOMContentLoaded', () => {
    const scanButton = document.getElementById('scan-button');
    const xrTemplate = document.getElementById('xr-template');
    const masterTemplate = document.getElementById('master-template');
    const deviceContainer = document.getElementById('device-container');
    const resultContainer = document.getElementById('resultContainer');

    // Handle "Scan" button click
    scanButton.addEventListener('click', () => {
        fetch('/scan')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to scan devices');
                }
                return response.json();
            })
            .then(responseData => {
                if (responseData.status !== "success") {
                    throw new Error(responseData.message);
                }
                deviceContainer.innerHTML = ''; // Clear existing devices
                addMasterNode(responseData.master); // Add master node
                if (responseData.nodes) {
                    Object.values(responseData.nodes).forEach(node => addNodePanel(node));
                }
            })
            .catch(error => {
                alert(`${error.message}`);
            });
    });

    function addMasterNode(masterInfo) {
        const clone = masterTemplate.content.cloneNode(true);
        const deviceHeader = clone.querySelector('.device-header');
        deviceHeader.setAttribute('data-name', masterInfo.name);
        deviceHeader.setAttribute('data-node-id', masterInfo.nodeID);
        deviceHeader.setAttribute('data-ip', masterInfo.addr.ip);
        deviceHeader.setAttribute('data-service-port', masterInfo.servicePort);
        clone.querySelector('.device-name').textContent = masterInfo.name;
        clone.querySelector('.device-type').textContent = `Type: Master`;
        clone.querySelector('.device-ip').textContent = `IP: ${masterInfo.addr.ip}`;
        deviceContainer.appendChild(clone);
    }

    function addNodePanel(nodeInfo) {
        const clone = xrTemplate.content.cloneNode(true);
        const deviceHeader = clone.querySelector('.device-header');
        deviceHeader.setAttribute('data-name', nodeInfo.name);
        deviceHeader.setAttribute('data-node-id', nodeInfo.nodeID);
        deviceHeader.setAttribute('data-ip', nodeInfo.addr.ip);
        deviceHeader.setAttribute('data-service-port', nodeInfo.servicePort);
        clone.querySelector('.device-name').textContent = nodeInfo.name;
        clone.querySelector('.device-type').textContent = `Type: ${nodeInfo.type}`;
        clone.querySelector('.device-ip').textContent = `IP: ${nodeInfo.addr.ip}`;
        addButtonEventListeners(clone, deviceHeader);
        deviceContainer.appendChild(clone);
    }

    function addButtonEventListeners(clone, deviceHeader) {
        const name = deviceHeader.getAttribute('data-name');
        const ip = deviceHeader.getAttribute('data-ip');
        const servicePort = deviceHeader.getAttribute('data-service-port');

        clone.querySelector('.start-btn').addEventListener('click', () => {
            sendPostRequest('/start-qr-alignment', { name, ip, servicePort }, 'Start QR Alignment');
        });

        clone.querySelector('.stop-btn').addEventListener('click', () => {
            sendPostRequest('/stop-qr-alignment', { name, ip, servicePort }, 'Stop QR Alignment');
        });

        clone.querySelector('.rename-btn').addEventListener('click', () => {
            const newName = prompt('Enter new name for the device:');
            sendPostRequest('/rename-device', { name, ip, servicePort, newName }, 'Rename Device');
            if (newName) {
                alert(`Device renamed to: ${newName}`);
            }
        });

        clone.querySelector('.view-btn').addEventListener('click', () => {
            alert(`Viewing capture for Node: ${deviceHeader.getAttribute('data-node-id')}`);
        });
        clone.querySelector('.env-occlusion-btn').addEventListener('click', () => {
            sendPostRequest('/env-occlusion', { name, ip, servicePort }, 'Environment Occlusion');
        });
    
        clone.querySelector('.log-btn').addEventListener('click', () => {
            alert(`Log requested for Node: ${deviceHeader.getAttribute('data-node-id')}`);
        });
    }

    function sendPostRequest(url, body, actionName) {
        fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        })
            .then(response => response.json())
            .catch(error => {
                console.error(`${actionName} Error:`, error);
                alert(`${actionName} failed.`);
            });
    }
});
