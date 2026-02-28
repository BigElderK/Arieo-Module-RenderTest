
#include "base/prerequisites.h"
#include "core/core.h"
#include "interface/rhi/rhi.h"
#include "interface/archive/archive.h"
#include "interface/file_loader/image_loader.h"
#include "interface/file_loader/model_loader.h"
#include "interface/main/main_module.h"

namespace Arieo
{
    Core::Coroutine::CorHandle<void> renderTestCoroutine()
    {
        Base::Interop::RawRef<Interface::Main::IMainModule> main_module = Core::ModuleManager::getInterface<Interface::Main::IMainModule>();
        Base::Interop::RawRef<Interface::Window::IWindowManager> window_manager = Core::ModuleManager::getInterface<Interface::Window::IWindowManager>();
        Base::Interop::RawRef<Interface::RHI::IRenderInstance> render_instance = Core::ModuleManager::getInterface<Interface::RHI::IRenderInstance>();
        Base::Interop::RawRef<Interface::FileLoader::IImageLoader> image_loader = Core::ModuleManager::getInterface<Interface::FileLoader::IImageLoader>();
        Base::Interop::RawRef<Interface::FileLoader::IModelLoader> model_loader = Core::ModuleManager::getInterface<Interface::FileLoader::IModelLoader>();

        co_yield Core::Coroutine::YieldUntil([&]() -> bool
        {
            Core::Logger::trace("Waiting for required interfaces...");
            return main_module != nullptr 
                && window_manager != nullptr 
                && render_instance != nullptr
                && image_loader != nullptr
                && model_loader != nullptr;
        });

        // initialize
        Base::Interop::RawRef<Interface::Window::IWindow> window = nullptr;
        Base::Interop::SharedRef<Interface::Archive::IArchive> content_archive = nullptr;

        Base::Interop::RawRef<Interface::RHI::IRenderDevice> render_device = nullptr;
        Base::Interop::RawRef<Interface::RHI::IRenderSurface> render_surface = nullptr;
        Base::Interop::RawRef<Interface::RHI::ISwapchain> render_swapchain = nullptr;
        Base::Interop::RawRef<Interface::RHI::IShader> test_vert_shader = nullptr;
        Base::Interop::RawRef<Interface::RHI::IShader> test_frag_shader = nullptr;
        Base::Interop::RawRef<Interface::RHI::IPipeline> render_pipline = nullptr;
        Base::Interop::RawRef<Interface::RHI::ICommandPool> command_pool = nullptr;
        Base::Interop::RawRef<Interface::RHI::IDescriptorPool> descriptor_pool = nullptr;
        Base::Interop::RawRef<Interface::RHI::IImage> texture_image = nullptr;
        Base::Interop::RawRef<Interface::RHI::IImage> depth_image = nullptr;
        {
            Core::Logger::trace("Getting main window");
            {
                if(window_manager == nullptr)
                {
                    Core::Logger::error("No window manager module found!");
                    co_return;
                }

                window = window_manager->getMainWindow();
                Core::Logger::trace("Main window got.");

                if(window == nullptr)
                {
                    Core::Logger::trace("Main window not found, creating new window");
                    window = window_manager->createWindow(0, 0, 1024, 768);
                }
            }

            Core::Logger::trace("Window size: {}x{}", window->getWindowRect().size.x, window->getWindowRect().size.y);

            Core::Logger::trace("creating surface");
            render_surface = render_instance->createSurface(window_manager, window);
            if(render_surface == nullptr)
            {
                Core::Logger::error("Create render surface failed!");
                co_return;
            }

            Core::Logger::trace("creating render_device");
            auto hardware_informations = render_instance->getHardwareInfomations();

            int selected_device_index = 0;
            for(int i = 0; i < hardware_informations.size(); i++)
            {
                if(hardware_informations[i].find("MultiViewport: Yes") != std::string::npos)
                {
                    selected_device_index = i;
                }
                Core::Logger::trace("Render Hardware Device found:\r\n{}", hardware_informations[i]);
            }

            render_device = render_instance->createDevice(selected_device_index, render_surface);
            if(render_device == nullptr)
            {
                Core::Logger::error("Create render device failed!");
                co_return;
            }

            Core::Logger::trace("creating swapchain");
            render_swapchain = render_device->createSwapchain(render_surface);
            if(render_swapchain == nullptr)
            {
                Core::Logger::error("Create swapchain failed!");
                co_return;
            }

            Core::Logger::trace("getting root archive");
            content_archive = main_module->getRootArchive();

            Core::Logger::trace("loading shaders");
            auto vert_shader_file = content_archive->aquireFileBuffer("content/shader/test_2.vert.hlsl.spv");
            auto frag_shader_file = content_archive->aquireFileBuffer("content/shader/test_2.frag.hlsl.spv");

            Core::Logger::trace("vert shader loaded: {}", vert_shader_file->getBufferSize());
            Core::Logger::trace("frag shader loaded: {}", frag_shader_file->getBufferSize());

            test_vert_shader = render_device->createShader(vert_shader_file->getBuffer(), vert_shader_file->getBufferSize());
            test_frag_shader = render_device->createShader(frag_shader_file->getBuffer(), frag_shader_file->getBufferSize());

            // content_archive->releaseFileBuffer(vert_shader_file);
            // content_archive->releaseFileBuffer(frag_shader_file);
        
            command_pool = render_device->getGraphicsCommandQueue()->createCommandPool();
            descriptor_pool = render_device->createDescriptorPool(10);
        }

        const size_t max_frames_in_flight = 3;


        // Loading model
        Core::Logger::trace("loading model");
        
        auto model_file = content_archive->aquireFileBuffer("content/model/viking_room.model.obj");
        auto model_buffer = model_loader->loadObj(model_file->getBuffer(), model_file->getBufferSize());
        

        // Create vertext buffer
        Core::Logger::trace("creating vertext buffer");
        Base::Interop::RawRef<Interface::RHI::IBuffer> vertex_buffer = render_device->createBuffer(
            sizeof(Interface::FileLoader::ModelVertex) * model_buffer->getVertexCount(),
            Interface::RHI::BufferUsageBitFlags::VERTEX | Interface::RHI::BufferUsageBitFlags::TRANSFER_DST, 
            Interface::RHI::BufferAllocationFlags::CREATE_DEDICATED_MEMORY_BIT,
            Interface::RHI::MemoryUsage::AUTO_PREFER_DEVICE
        );

        // Create index buffer
        Core::Logger::trace("creating index buffer");
        Base::Interop::RawRef<Interface::RHI::IBuffer> index_buffer = render_device->createBuffer(
            sizeof(Interface::FileLoader::ModelVertex) * model_buffer->getVertexCount(),
            Interface::RHI::BufferUsageBitFlags::INDEX | Interface::RHI::BufferUsageBitFlags::TRANSFER_DST, 
            Interface::RHI::BufferAllocationFlags::CREATE_DEDICATED_MEMORY_BIT | Interface::RHI::BufferAllocationFlags::CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
            Interface::RHI::MemoryUsage::AUTO_PREFER_DEVICE
        );

        Core::Logger::trace("creating staging buffer");
        Base::Interop::RawRef<Interface::RHI::IBuffer> staging_vertext_buffer = render_device->createBuffer(
            sizeof(Interface::FileLoader::ModelVertex) * model_buffer->getVertexCount(),
            Interface::RHI::BufferUsageBitFlags::TRANSFER_SRC, 
            Interface::RHI::BufferAllocationFlags::CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, 
            Interface::RHI::MemoryUsage::AUTO_PREFER_HOST
        );

        Core::Logger::trace("vertext memory copying start");
        void* mapped_staging_vertext_mem = staging_vertext_buffer->mapMemory(0, 0);
        memcpy(mapped_staging_vertext_mem, model_buffer->getVertices(), sizeof(Interface::FileLoader::ModelVertex) * model_buffer->getVertexCount());
        staging_vertext_buffer->unmapMemory();
        Core::Logger::trace("vertext memory copying end");

        Core::Logger::trace("index memory copying start");
        void* mapped_index_vertext_mem = index_buffer->mapMemory(0, 0);
        memcpy(mapped_index_vertext_mem, model_buffer->getIndices(), sizeof(uint16_t) * model_buffer->getIndexCount());
        index_buffer->unmapMemory();
        Core::Logger::trace("index memory copying end");

        size_t model_vertex_count = model_buffer->getVertexCount();
        size_t model_index_count = model_buffer->getIndexCount();
        model_loader->unloadObj(model_buffer);

        Core::Logger::trace("loading texture image");
        auto image_file = content_archive->aquireFileBuffer("content/model/viking_room.dds");
        Base::Interop::SharedRef<Interface::FileLoader::IImageBuffer> image_buffer = image_loader->loadDDS(image_file);
        Core::Logger::trace("Texture file loaded {} {} {}", image_buffer->getWidth(), image_buffer->getHeight(), (std::uint32_t)image_buffer->getFormat());

        Core::Logger::trace("creating texture image");
        texture_image = render_device->createImage(
            image_buffer->getWidth(), image_buffer->getHeight(), 
            image_buffer->getFormat(),
            Interface::RHI::ImageAspectFlags::COLOR_BIT,
            Interface::RHI::ImageTiling::OPTIMAL,
            Interface::RHI::ImageUsageFlags::SAMPLED_BIT | Interface::RHI::ImageUsageFlags::TRANSFER_DST_BIT,
            Interface::RHI::MemoryUsage::AUTO_PREFER_DEVICE
        );
        Core::Logger::trace("texture image created with memory capacity: {}", texture_image->getMemorySize());

        Core::Logger::trace("creating depth image");
        Interface::RHI::Format depth_format = render_device->findSupportedFormat(
            {Interface::RHI::Format::D32_SFLOAT, Interface::RHI::Format::D32_SFLOAT_S8_UINT, Interface::RHI::Format::D24_UNORM_S8_UINT},
            Interface::RHI::ImageTiling::OPTIMAL,
            Interface::RHI::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT_BIT
        );
        if(depth_format == Interface::RHI::Format::UNKNOWN)
        {
            Core::Logger::error("Cannot found format for depth image");
            co_return;
        }
        Core::Logger::trace("depth format selected: {}", (std::uint32_t)depth_format);
        depth_image = render_device->createImage(
            render_swapchain->getExtent().size.x, render_swapchain->getExtent().size.y,
            depth_format,
            Interface::RHI::ImageAspectFlags::DEPTH_BIT,
            Interface::RHI::ImageTiling::OPTIMAL,
            Interface::RHI::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT_BIT,
            Interface::RHI::MemoryUsage::AUTO_PREFER_DEVICE
        );

        Core::Logger::trace("creating texture stage buffer");
        Base::Interop::RawRef<Interface::RHI::IBuffer> staging_image_buffer = render_device->createBuffer(
            image_buffer->getSize(),
            Interface::RHI::BufferUsageBitFlags::TRANSFER_SRC, 
            Interface::RHI::BufferAllocationFlags::CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT, 
            Interface::RHI::MemoryUsage::AUTO_PREFER_HOST
        );
        void* staging_image_staged_buffer = staging_image_buffer->mapMemory(0, image_buffer->getSize());
        memcpy(staging_image_staged_buffer, image_buffer->getBuffer(), image_buffer->getSize());
        staging_image_buffer->unmapMemory();
        // content_archive->releaseFileBuffer(image_file);

        Base::Interop::RawRef<Interface::RHI::ICommandBuffer> image_copy_command_buffer = command_pool->allocateCommandBuffer();
        image_copy_command_buffer->begin();
        image_copy_command_buffer->copyBufferToImage(staging_image_buffer, texture_image);
        image_copy_command_buffer->prepareDepthImage(depth_image);
        image_copy_command_buffer->end();
        render_device->getGraphicsCommandQueue()->submitCommand(image_copy_command_buffer);
        render_device->getGraphicsCommandQueue()->waitIdle();

        // Create pipline
        //TODO:
        render_pipline = render_device->createPipeline(
            test_vert_shader, 
            test_frag_shader, 
            render_swapchain->getImageViews()[0],
            depth_image->getImageView()
        );

        // Create framebuffer
        std::vector<Base::Interop::RawRef<Interface::RHI::IFramebuffer>> framebuffer_array;
        for(Base::Interop::RawRef<Interface::RHI::IImageView> swapchain_image_view : render_swapchain->getImageViews())
        {
            std::vector<Base::Interop::RawRef<Interface::RHI::IImageView>> image_views{swapchain_image_view, depth_image->getImageView()};
            framebuffer_array.emplace_back(
                render_device->createFramebuffer(
                    render_pipline, 
                    render_swapchain, 
                    image_views
                )
            );
        }

        // FrameContext
        class FrameContext
        {
        public:
            struct UniformBufferObject 
            {
                Base::Math::Matrix4 m_model;
                Base::Math::Matrix4 m_view;
                Base::Math::Matrix4 m_proj;
            };

            UniformBufferObject m_uniform_obj;
            Base::Interop::RawRef<Interface::RHI::IDescriptorSet> m_descriptor_set = nullptr;

            Base::Interop::RawRef<Interface::RHI::IBuffer> m_uniform_buffer = nullptr;
            Base::Interop::RawRef<Interface::RHI::IFence> m_fence = nullptr;
            Base::Interop::RawRef<Interface::RHI::ISemaphore> m_image_availiable_semaphore = nullptr;
            Base::Interop::RawRef<Interface::RHI::ISemaphore> m_render_finished_semaphore = nullptr;        
            Base::Interop::RawRef<Interface::RHI::ICommandBuffer> m_command_buffer = nullptr;
        };
        std::vector<FrameContext> frame_context_array(max_frames_in_flight);

        for(FrameContext& frame_context : frame_context_array)
        {
            frame_context.m_fence = render_device->createFence();
            frame_context.m_image_availiable_semaphore = render_device->createSemaphore();
            frame_context.m_render_finished_semaphore = render_device->createSemaphore();
            frame_context.m_command_buffer = command_pool->allocateCommandBuffer();
            frame_context.m_descriptor_set = descriptor_pool->allocateDescriptorSet(render_pipline);
            frame_context.m_uniform_buffer = render_device->createBuffer(
                sizeof(FrameContext::UniformBufferObject),
                Interface::RHI::BufferUsageBitFlags::UNIFORM,
                Interface::RHI::BufferAllocationFlags::CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                Interface::RHI::MemoryUsage::AUTO_PREFER_DEVICE
            );
        }

        auto start = std::chrono::high_resolution_clock::now();
        
        // ticking
        {
            std::uint32_t frame_count = 0;
            std::uint32_t frame_index = 0;
            while(window->isClosed() == false)
            {   
                co_yield std::suspend_always{};
                frame_count++;
                frame_index = frame_count % max_frames_in_flight; 
                FrameContext& frame_context = frame_context_array[frame_index];

                // window_manager->tick();
                // std::this_thread::sleep_for(std::chrono::seconds(1));
                auto end = std::chrono::high_resolution_clock::now();
                float duration = std::chrono::duration<float>(end - start).count();

                // Draw frame
                {
                    frame_context.m_fence->wait();
                    frame_context.m_fence->reset();

                    std::uint32_t cur_framebuffer_index = std::numeric_limits<std::uint32_t>::max();
                    while(cur_framebuffer_index == std::numeric_limits<std::uint32_t>::max())
                    {
                        if(render_swapchain->isLost())
                        {
                            while(window->getWindowRect().size.x == 0 || window->getWindowRect().size.y == 0)
                            {
                                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                            }

                            render_device->waitIdle();
                            render_device->destroySwapchain(render_swapchain);
                            render_device->waitIdle();
                            render_swapchain = render_device->createSwapchain(render_surface);

                            for(Base::Interop::RawRef<Interface::RHI::IFramebuffer> frame_buffer : framebuffer_array)
                            {
                                render_device->destroyFramebuffer(frame_buffer);
                            }
                            framebuffer_array.clear();
                            for(Base::Interop::RawRef<Interface::RHI::IImageView> swapchain_image_view : render_swapchain->getImageViews())
                            {
                                std::vector<Base::Interop::RawRef<Interface::RHI::IImageView>> image_views{swapchain_image_view, depth_image->getImageView()};
                                framebuffer_array.emplace_back(
                                    render_device->createFramebuffer(
                                        render_pipline, 
                                        render_swapchain, 
                                        image_views
                                    )
                                );
                            }
                        }
                        cur_framebuffer_index = render_swapchain->acquireNextImageIndex(frame_context.m_image_availiable_semaphore);
                    }
                    
                    // update uniform data
                    {
                        // update uniform_obj
                        // frame_context.m_uniform_obj.m_model 
                        // frame_context.m_uniform_obj.m_model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
                        // frame_context.m_uniform_obj.m_view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
                        // frame_context.m_uniform_obj.m_proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float) swapChainExtent.height, 0.1f, 10.0f);
                        // frame_context.m_uniform_obj[1][1] *= -1;
                        frame_context.m_uniform_obj.m_model = 
                            Base::Math::Quaternion::FromAngleAxis(
                            Base::Math::radians(duration * 180.0f), 
                            Base::Math::Vector3(0.0f, 1.0f, 0.0f)).ToMatrix4();
                        //frame_context.m_uniform_obj.m_model = Base::Math::Matrix4<float>::FromTranslationVector(Base::Math::Vector3<float>(0.0f,  0.0f, 0.0f));

                        frame_context.m_uniform_obj.m_view = 
                            Base::Math::Matrix4::LookAt(
                                Base::Math::Vector3(0.0f, 0.0f, 0.0f), // at
                                Base::Math::Vector3(2.0f, 1.0f, 2.0f), // eye
                                Base::Math::Vector3(0.0f, 1.0f, 0.0f) // up
                            );

                        frame_context.m_uniform_obj.m_proj = 
                        Base::Math::Matrix4::Perspective(
                            Base::Math::radians(45.0f),
                            window->getWindowRect().size.x / (float) window->getWindowRect().size.y,
                            0.1f,
                            1000.0f
                        ) * Base::Math::Coordinate::getNDCMatrix();

                        /*
                        {
                            Base::Math::Vector3<float> result;
                            result = frame_context.m_uniform_obj.m_proj * frame_context.m_uniform_obj.m_view * frame_context.m_uniform_obj.m_model * Base::Math::Vector3<float>(-1.0f, 0.0f, -1.0f);
                            Core::Logger::trace("result: {} {} {}", result.x, result.y, result.z);
                            result = frame_context.m_uniform_obj.m_proj * frame_context.m_uniform_obj.m_view * frame_context.m_uniform_obj.m_model * Base::Math::Vector3<float>(1.0f, 0.0f, -1.0f);
                            Core::Logger::trace("result: {} {} {}", result.x, result.y, result.z);
                            result = frame_context.m_uniform_obj.m_proj * frame_context.m_uniform_obj.m_view * frame_context.m_uniform_obj.m_model * Base::Math::Vector3<float>(1.0f, 0.0f, 1.0f);
                            Core::Logger::trace("result: {} {} {}", result.x, result.y, result.z);
                        }
                        */
                        
                        void* mapped_memory = frame_context.m_uniform_buffer->mapMemory(0, 0);
                        memcpy(mapped_memory, &frame_context.m_uniform_obj, sizeof(frame_context.m_uniform_obj));
                        frame_context.m_uniform_buffer->unmapMemory();

                        frame_context.m_descriptor_set->bindBuffer(0, frame_context.m_uniform_buffer, 0, sizeof(frame_context.m_uniform_obj));
                        frame_context.m_descriptor_set->bindImage(1, texture_image);
                    }

                    Core::Logger::trace("framecount: wokao #3 {}", frame_count);

                    // Record command buffer
                    frame_context.m_command_buffer->reset();
                    {
                        frame_context.m_command_buffer->begin();
                        
                        frame_context.m_command_buffer->copyBuffer(staging_vertext_buffer, vertex_buffer, sizeof(Interface::FileLoader::ModelVertex) * model_vertex_count);

                        frame_context.m_command_buffer->beginRenderPass(render_pipline, framebuffer_array[cur_framebuffer_index]);
                        frame_context.m_command_buffer->bindPipeline(render_pipline);
                        frame_context.m_command_buffer->bindVertexBuffer(vertex_buffer, 0);
                        frame_context.m_command_buffer->bindIndexBuffer(index_buffer, 0);

                        frame_context.m_command_buffer->bindDescriptorSets(render_pipline, frame_context.m_descriptor_set);

                        //frame_context.m_command_buffer->draw(static_cast<uint32_t>(vertices.size()), 1, 0, 0);
                        frame_context.m_command_buffer->drawIndexed(static_cast<uint32_t>(model_index_count), 1, 0, 0, 0);
                        
                        frame_context.m_command_buffer->endRenderPass();
                        frame_context.m_command_buffer->end();
                    }

                    // Submit command buffer
                    render_device->getGraphicsCommandQueue()->submitCommand(
                        frame_context.m_command_buffer, 
                        frame_context.m_fence, 
                        frame_context.m_image_availiable_semaphore, 
                        frame_context.m_render_finished_semaphore);

                    // Present
                    render_device->getPresentCommandQueue()->present(render_swapchain, cur_framebuffer_index, framebuffer_array[cur_framebuffer_index], frame_context.m_render_finished_semaphore);
                }
            }
        }

        // Clear
        render_device->waitIdle();
        render_device->destroyBuffer(vertex_buffer);
        render_device->destroyBuffer(index_buffer);
        render_device->destroyBuffer(staging_vertext_buffer);
        render_device->destroyBuffer(staging_image_buffer);

        command_pool->freeCommandBuffer(image_copy_command_buffer);
        
        for(FrameContext& frame_context : frame_context_array)
        {
            render_device->destroyBuffer(frame_context.m_uniform_buffer);
            render_device->destroyFence(frame_context.m_fence);
            render_device->destroySemaphore(frame_context.m_image_availiable_semaphore);
            render_device->destroySemaphore(frame_context.m_render_finished_semaphore);
            command_pool->freeCommandBuffer(frame_context.m_command_buffer);
        }

        // finalilze
        {
            for(Base::Interop::RawRef<Interface::RHI::IFramebuffer> frame_buffer : framebuffer_array)
            {
                render_device->destroyFramebuffer(frame_buffer);
            }
            framebuffer_array.clear();

            window_manager->destroyWindow(window);

            render_device->destroyDescriptorPool(descriptor_pool);
            render_device->getGraphicsCommandQueue()->destroyCommandPool(command_pool);
            render_device->destroyPipeline(render_pipline);
            render_device->destroyShader(test_vert_shader);
            render_device->destroyShader(test_frag_shader);
            render_device->destroyImage(texture_image);
            render_device->destroyImage(depth_image);

            render_device->destroySwapchain(render_swapchain);
            
            render_instance->destroyDevice(render_device);
            render_instance->destroySurface(render_surface);
        }
        co_return;
    }

    Core::Coroutine::CorHandle<void> basicTestCoroutine(int i)
    {
        Core::Logger::trace("step_1");
        co_yield std::suspend_always{};   

        Core::Logger::trace("step_1.1");
        co_yield Core::Coroutine::YieldUntil([]() -> bool
        {
            return true;
        });
        
        Core::Logger::trace("step_2");
        co_yield std::suspend_always{};    
        Core::Logger::trace("step_3");
        
        // no pararell task.
        // std::uint32_t ret = 
        co_yield Core::Coroutine::YieldSubCoroutine([]() -> Core::Coroutine::CorHandle<std::uint32_t>
        {
            co_yield std::suspend_always{};    
            Core::Logger::trace("step_3.1");
            co_yield std::suspend_always{};    
            Core::Logger::trace("step_3.2");
            
            int ret = co_yield Core::Coroutine::YieldSubCoroutine([]() -> Core::Coroutine::CorHandle<int>
            {
                co_yield std::suspend_always{}; 
                Core::Logger::trace("step_3.2.1");
                co_yield std::suspend_always{}; 
                Core::Logger::trace("step_3.2.2");
                co_yield std::suspend_always{}; 
                Core::Logger::trace("step_3.2.3");
                co_return 10;
            }());

            // Core::Logger::trace("step_3.3");
            co_return 1 + ret;
        }());

        // Core::Logger::trace("step_3 return {}", ret);

        std::uint64_t ret_value = co_yield Core::Coroutine::YieldUpdateOnce<std::uint64_t>(
            [](Core::Coroutine::Task& running_task) -> std::uint64_t
            {
                return 123;
            }
        );

        // Core::Logger::trace("step_4 return {}", ret_value);

        while(true)
        {
            size_t concurrent_queue_num = 1000;
            Base::ConcurrentQueue<int> result_queue;
            for(int i = 0; i < concurrent_queue_num; i++)
            {
                co_yield Core::Coroutine::CreateParallelCoroutine(
                    [](int i, Base::ConcurrentQueue<int>& result_queue) -> Core::Coroutine::CorHandle<void>
                    {
                        co_yield std::suspend_always{};    
                        // Core::Logger::trace("parallel {} start", i);
                        co_yield Core::Coroutine::YieldSubCoroutine([]() -> Core::Coroutine::CorHandle<void>
                        {
                            co_yield std::suspend_always{}; 
                            // Core::Logger::trace("parallel step_0");
                            co_yield std::suspend_always{}; 
                            // Core::Logger::trace("parallel step_1");
                            co_yield std::suspend_always{}; 
                            // Core::Logger::trace("parallel step_2");
                            co_return;
                        }());

                        co_yield std::suspend_always{};    
                        // Core::Logger::trace("parallel {} end", i);

                        result_queue.enqueue(Core::ThreadPool::getCurrentThreadId());
                        // Core::Logger::trace("parallel {} result enqueued: {}", i, Core::ThreadPool::getCurrentThreadId());
                        co_return;
                    }(i, result_queue)
                );
            }

            co_yield Core::Coroutine::YieldUntil(
                [&result_queue, concurrent_queue_num]() -> bool
                {
                    return result_queue.size_approx() == concurrent_queue_num;
                }
            );
            
            Core::Logger::trace("wait parallel done");
        }


        Core::Logger::trace("step_5");
        co_return;
    };

    GENERATOR_MODULE_ENTRY_FUN()
    ARIEO_DLLEXPORT void ModuleMain()
    {
        Core::Logger::setDefaultLogger("render_test_module");
        static struct DllLoader
        {
            DllLoader()
            {
                Base::Interop::RawRef<Interface::Main::IMainModule> main_module = Core::ModuleManager::getInterface<Interface::Main::IMainModule>();
                main_module->enqueueTask(
                    Core::Coroutine::Task::generatorTasklet(renderTestCoroutine())
                );
            }

            ~DllLoader()
            {
            }
        } dll_loader;
    }
}




